// SPDX-License-Identifier: GPL-2.0-or-later
// TLG_VKB_SHADER_PREBUILD: the new engine's background shader pre-build from the shipped list (see the header)

#include "native/native_vkb_shader_prebuild.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

#include "common/logging/log.h"
#include "native/native_vkb_pipeline_warm.h"
#include "native/native_vkb_shader_list.h"
#include "native/native_vkb_shader_recipe.h"
#include "native/native_vkb_shader_store.h"
#include "video_core/renderer_vulkan/shader_list.h" // the normal renderer's list reader (read only)
#include "tlg_overlay.h"

namespace Core::Vkb::Prebuild {
namespace {

/// Lives as long as the process: the build's threads may outlive the game's shutdown
struct State {
    std::mutex mutex;
    std::condition_variable done;
    bool running = false;
    std::chrono::steady_clock::time_point started{};
    std::vector<u64> listed; ///< sorted
    u32 waits_given_up = 0;
    // the last build's numbers (Report)
    std::string list, outcome = "not started";
    u64 entries = 0, built = 0, already = 0, skipped = 0, write_failed = 0, restarts = 0, parts = 0, parts_failed = 0, accounted = 0, ms = 0, rescanned = 0;
    u64 fixed_dx10_clamp = 0, fixed_depth_before_shader = 0; ///< way A: values the conversion fixed (native_vkb_shader_list.h)
    std::map<std::string, u64> reasons;
};
State& S() {
    static State* const state = new State;
    return *state;
}

long EnvSeconds(const char* name, long fallback, long low, long high) {
    if (const char* env = std::getenv(name); env != nullptr && *env != '\0') {
        char* end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end != env && *end == '\0' && value >= low && value <= high) {
            return value;
        }
        LOG_WARNING(Render_Vulkan, "{}={}: not a number from {} to {}; {} is used", name, env, low, high, fallback);
    }
    return fallback;
}

std::atomic<u32> unguarded{0}; ///< Windows: build processes started without the game's job object (they may outlive the game)

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream f{path, std::ios::binary};
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::string FirstLine(const std::filesystem::path& path) {
    std::ifstream f{path};
    std::string line;
    std::getline(f, line);
    return line;
}

struct ChildEnd {
    int code = -1;      ///< exit code; -1 when it did not exit by itself or did not start
    int signal = 0;     ///< Linux: the signal that ended it
    bool hung = false;  ///< stopped: no progress for the entry limit
    bool slow = false;  ///< stopped: ran longer than the part limit
    bool started = true;
};

/// One run of a build process (the normal renderer's RunChildOnce, video_core/renderer_vulkan/shader_recipe.cpp, copied: it is internal to
/// that file; the engine uses only the normal renderer's game-file reader, through its header, native_vkb_shader_list.cpp). Stopped when its progress file names the same entry for longer than `entry_s`, or when it runs longer
/// than `part_s`.
/// How far each build process is (the second number of its progress file: entries of its part reached), for the corner note
/// "Building shaders NN%", as the normal renderer's build shows it.
std::array<std::atomic<u32>, 32> part_reached{};
std::atomic<u32> build_entries{0};

void ShowProgress() {
    u32 reached = 0;
    for (const auto& n : part_reached) {
        reached += n.load(std::memory_order_relaxed);
    }
    TlgOverlay::SetShaderBuild(reached, std::max(build_entries.load(std::memory_order_relaxed), 1u));
}

ChildEnd RunOnce(const std::vector<std::string>& args, const std::filesystem::path& progress, long entry_s, long part_s, u32 part) {
    ChildEnd end;
    std::string last;
    const auto t0 = std::chrono::steady_clock::now();
    auto changed = t0;
    const auto stop = [&] {
        const auto now = std::chrono::steady_clock::now();
        if (now - t0 > std::chrono::seconds{part_s}) {
            end.slow = true;
            return true;
        }
        if (auto text = ReadText(progress); text != last) {
            last = std::move(text);
            changed = now;
            unsigned long long index = 0;
            unsigned reached = 0;
            if (std::sscanf(last.c_str(), "%llu %u", &index, &reached) == 2) {
                part_reached[part].store(reached, std::memory_order_relaxed);
                ShowProgress();
            }
            return false;
        }
        if (!last.empty() && now - changed > std::chrono::seconds{entry_s}) {
            end.hung = true;
            return true;
        }
        return false;
    };
#ifdef _WIN32
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring command = L"\"" + std::wstring{exe} + L"\" --vkb-build-store";
    for (const auto& arg : args) {
        command += L" \"" + std::filesystem::u8path(arg).wstring() + L"\"";
    }
    // the game's job object: its build processes end with the game (the job closes with the game's last handle)
    static const HANDLE job = [] {
        HANDLE h = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (h == nullptr || !SetInformationJobObject(h, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
            std::printf("Shader prebuild (new engine): no job object (error %lu): build processes may outlive the game\n", GetLastError());
            unguarded.fetch_add(1, std::memory_order_relaxed);
            return HANDLE{nullptr};
        }
        return h;
    }();
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi)) {
        end.started = false;
        return end;
    }
    if (job != nullptr && !AssignProcessToJobObject(job, pi.hProcess)) {
        std::printf("Shader prebuild (new engine): a build process is not in the game's job object (error %lu): it may outlive the game\n", GetLastError());
        unguarded.fetch_add(1, std::memory_order_relaxed);
    }
    ResumeThread(pi.hThread);
    for (;;) {
        if (WaitForSingleObject(pi.hProcess, 100) != WAIT_TIMEOUT) {
            break;
        }
        if (stop()) {
            TerminateProcess(pi.hProcess, 1);
            WaitForSingleObject(pi.hProcess, INFINITE);
            break;
        }
    }
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    end.code = end.hung || end.slow ? -1 : static_cast<int>(exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
#else
    std::vector<std::string> strings{"tlg-probe", "--vkb-build-store"};
    strings.insert(strings.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& s : strings) {
        argv.push_back(s.data());
    }
    argv.push_back(nullptr);
    std::fflush(stdout);
    // posix_spawn, not fork: the game runs meanwhile, and a fork copies its (large) memory map while every other thread of the game stops
    // its own process group: a stop signal sent to the game's group (a terminal's Ctrl+C, a test harness) does not reach it; it ends with the
    // game through its parent-death signal (native_vkb_shader_recipe.cpp, BuildList)
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attr, 0);
    pid_t pid = 0;
    const int spawned = posix_spawn(&pid, "/proc/self/exe", nullptr, &attr, argv.data(), environ);
    posix_spawnattr_destroy(&attr);
    if (spawned != 0) {
        end.started = false;
        return end;
    }
    int status = 0;
    bool stopped = false;
    for (;;) {
        const pid_t done = waitpid(pid, &status, WNOHANG);
        if (done == pid || (done < 0 && errno != EINTR)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        if (stop()) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
            }
            stopped = true;
            break;
        }
    }
    if (!stopped) {
        end.code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        end.signal = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
    }
#endif
    return end;
}

std::mutex left_out_mutex;

struct PartEnd {
    int code = -1;
    u32 restarts = 0;
    std::string how; ///< empty: ended well
};

/// One part of the build until it ends well: a process that stops on an entry is started again without it (written to the left-out file),
/// at most 32 times; the stages it built before are in the folder and are not built again.
PartEnd RunPart(const std::filesystem::path& store, const std::filesystem::path& list, const std::filesystem::path& game, u32 part, u32 parts) {
    const long entry_s = EnvSeconds("TLG_VKB_SHADER_PREBUILD_ENTRY_S", 30, 1, 3600);
    const long part_s = EnvSeconds("TLG_VKB_SHADER_PREBUILD_PART_S", 600, 1, 86400);
    const auto progress = ShaderList::ProgressFile(store, part);
    const std::vector<std::string> args{store.string(), "--list", list.string(), game.string(), std::to_string(part), std::to_string(parts)};
    PartEnd result;
    std::error_code ec;
    for (u32 runs = 1;; ++runs) {
        std::filesystem::remove(progress, ec);
        const auto end = RunOnce(args, progress, entry_s, part_s, part);
        const auto at = ReadText(progress);
        std::filesystem::remove(progress, ec);
        result.code = end.code;
        if (end.code == 0) {
            result.how.clear();
            return result;
        }
#ifndef _WIN32 // Windows: the game's job object ends its build processes with it; there are no signals
        if (end.signal == SIGTERM || end.signal == SIGINT || end.signal == SIGHUP) { // stopped from outside (the game or the machine shuts down)
            result.how = fmt::format("stopped from outside (signal {})", end.signal);
            std::printf("Shader prebuild (new engine): build process %u was %s; its part ends here, the next start goes on with it\n", part, result.how.c_str());
            LOG_WARNING(Render_Vulkan, "vkb shader pre-build: build process {} was {}", part, result.how);
            return result;
        }
#endif
        if (end.slow) { // the whole part was too slow (a slow machine), not one entry: nothing is blamed, the part ends here (counted)
            result.how = fmt::format("it ran longer than {} s (TLG_VKB_SHADER_PREBUILD_PART_S)", part_s);
            std::printf("Shader prebuild (new engine): build process %u stopped: %s; its part ends here, the next start goes on with it\n", part, result.how.c_str());
            LOG_WARNING(Render_Vulkan, "vkb shader pre-build: build process {} stopped: {}", part, result.how);
            return result;
        }
        result.how = !end.started ? "it could not be started"
                     : end.hung   ? fmt::format("no progress for {} s on one entry", entry_s)
                     : end.signal ? fmt::format("signal {}", end.signal)
                                  : fmt::format("exit {}", end.code);
        // the progress names an entry only while the process builds one: a process that stops before its first entry or after its last is
        // not started again
        unsigned long long index = 0;
        const bool named = std::sscanf(at.c_str(), "%llu", &index) == 1;
        if (!named || runs > 32) {
            std::printf("Shader prebuild (new engine): build process %u stopped (%s)%s; its part of the build ends here\n", part, result.how.c_str(),
                        named ? ", after 32 restarts" : "");
            LOG_WARNING(Render_Vulkan, "vkb shader pre-build: build process {} stopped ({}); its part ends here", part, result.how);
            return result;
        }
        {
            std::scoped_lock lock{left_out_mutex};
            std::ofstream out{ShaderList::LeftOutFile(store), std::ios::binary | std::ios::app};
            out << index << " " << result.how << "\n";
        }
        ++result.restarts;
        std::printf("Shader prebuild (new engine): build process %u stopped (%s) on list entry %llu; that entry is left out (list-build-left-out.txt) and the "
                    "build goes on\n",
                    part, result.how.c_str(), index);
        LOG_WARNING(Render_Vulkan, "vkb shader pre-build: build process {} stopped ({}) on list entry {}; left out, started again", part, result.how, index);
    }
}

void Finish() {
    State& st = S();
    {
        std::scoped_lock lock{st.mutex};
        st.running = false;
    }
    st.done.notify_all();
}

void Run(const std::filesystem::path& store, const std::filesystem::path& list_path, const std::filesystem::path& game) {
    State& st = S();
    const auto t0 = std::chrono::steady_clock::now();
    std::string error;
    const auto list = Vulkan::ShaderList::Read(list_path, error); // the normal renderer's list and reader (native_vkb_shader_list.h)
    if (!list) {
        std::printf("Shader prebuild (new engine): shader list %s not used: %s\n", list_path.string().c_str(), error.c_str());
        LOG_WARNING(Render_Vulkan, "vkb shader pre-build: shader list {} not used: {}", list_path.string(), error);
        {
            std::scoped_lock lock{st.mutex};
            st.outcome = "list not read: " + error;
        }
        Finish();
        return;
    }
    PipelineWarm::NoteList(*list); // the list's pipeline keys for the warm-up check (native_vkb_pipeline_warm.h)
    {
        std::vector<u64> listed;
        for (const auto& item : list->entries) {
            listed.push_back(item.pgm_hash);
        }
        std::ranges::sort(listed);
        listed.erase(std::unique(listed.begin(), listed.end()), listed.end());
        std::scoped_lock lock{st.mutex};
        st.listed = std::move(listed);
        st.entries = list->entries.size();
    }
    const auto list_line = fmt::format("list {:016x}", ShaderList::FileId(list_path));
    // the game files the programs are read from: an updated game builds again
    std::string game_files;
    for (const char* name : {"trico_bootp.psarc", "trico_boot.psarc", "eboot.bin"}) {
        std::error_code size_ec, time_ec;
        const auto size = std::filesystem::file_size(game / name, size_ec);
        const auto time = std::filesystem::last_write_time(game / name, time_ec);
        game_files += fmt::format(" {}:{}:{}", name, size_ec ? 0 : size, time_ec ? 0 : static_cast<long long>(time.time_since_epoch().count()));
    }
    const auto marker_line = fmt::format("{} game{}", list_line, game_files);
    if (FirstLine(ShaderList::BuiltMarker(store)) == marker_line) {
        // the fixed values of that build stay in the report (the marker keeps them)
        u64 fixed_dx10_clamp = 0, fixed_depth_before_shader = 0;
        {
            std::ifstream f{ShaderList::BuiltMarker(store)};
            std::string word;
            std::getline(f, word);
            while (f >> word) {
                u64 n = 0;
                if (word == "fixed_dx10_clamp" && f >> n) {
                    fixed_dx10_clamp = n;
                } else if (word == "fixed_depth_before_shader" && f >> n) {
                    fixed_depth_before_shader = n;
                }
            }
        }
        std::printf("Shader prebuild (new engine): the list (%zu entries) was built into this store before (list-built.txt): nothing to build; fixed values set "
                    "then: dx10_clamp %llu, depth_before_shader %llu\n",
                    list->entries.size(), static_cast<unsigned long long>(fixed_dx10_clamp), static_cast<unsigned long long>(fixed_depth_before_shader));
        {
            std::scoped_lock lock{st.mutex};
            st.outcome = "built before";
            st.fixed_dx10_clamp = fixed_dx10_clamp;
            st.fixed_depth_before_shader = fixed_depth_before_shader;
        }
        Finish();
        PipelineWarm::Start(); // the pipelines from the stored stages (native_vkb_pipeline_warm.h)
        return;
    }
    std::error_code ec;
    std::filesystem::remove(ShaderList::BuiltMarker(store), ec);
    if (FirstLine(ShaderList::LeftOutFile(store)) != list_line) { // entries left out by an earlier build of another list do not count
        std::ofstream f{ShaderList::LeftOutFile(store), std::ios::trunc};
        f << list_line << "\n";
    }
    u32 jobs = std::clamp(std::max(1u, std::thread::hardware_concurrency()) / 2, 1u, 8u);
    if (const char* env = std::getenv("TLG_VKB_SHADER_PREBUILD_JOBS"); env != nullptr && std::atoi(env) > 0) {
        jobs = static_cast<u32>(std::min(std::atoi(env), 32));
    }
    for (u32 part = 0; part < 32; ++part) {
        std::filesystem::remove(ShaderList::ResultFile(store, part), ec);
    }
    std::printf("Shader prebuild (new engine): building the shader list in the background: %zu entries, %u build processes; list %s\n", list->entries.size(), jobs,
                list_path.string().c_str());
    // test only: TLG_VKB_SHADER_PREBUILD_DELAY_MS holds the build back, to check the waits
    if (const char* env = std::getenv("TLG_VKB_SHADER_PREBUILD_DELAY_MS"); env != nullptr && *env != '\0') {
        std::this_thread::sleep_for(std::chrono::milliseconds{std::atoi(env)});
    }
    std::vector<PartEnd> ends(jobs);
    build_entries.store(static_cast<u32>(list->entries.size()), std::memory_order_relaxed);
    ShowProgress(); // the corner note shows from the start of the build to its end
    {
        std::vector<std::thread> threads;
        for (u32 part = 0; part < jobs; ++part) {
            threads.emplace_back([&, part] { ends[part] = RunPart(store, list_path, game, part, jobs); });
        }
        for (auto& t : threads) {
            t.join();
        }
    }
    TlgOverlay::SetShaderBuild(0, 0);
    // every entry ends counted: the parts' results, and the parts that left none
    u64 entries = 0, built = 0, already = 0, skipped = 0, write_failed = 0, restarts = 0, failed = 0, fixed_dx10_clamp = 0, fixed_depth_before_shader = 0;
    std::map<std::string, u64> reasons;
    for (u32 part = 0; part < jobs; ++part) {
        restarts += ends[part].restarts;
        failed += !ends[part].how.empty();
        std::ifstream f{ShaderList::ResultFile(store, part)};
        if (!f) {
            continue;
        }
        std::string word;
        while (f >> word) {
            u64 n = 0;
            f >> n;
            if (word == "entries") {
                entries += n;
            } else if (word == "fixed_dx10_clamp") {
                fixed_dx10_clamp += n;
            } else if (word == "fixed_depth_before_shader") {
                fixed_depth_before_shader += n;
            } else if (word == "built") {
                built += n;
            } else if (word == "already") {
                already += n;
            } else if (word == "write_failed") {
                write_failed += n;
            } else if (word == "skipped") {
                std::string reason;
                std::getline(f, reason);
                reasons[reason.empty() ? "?" : reason.substr(1)] += n;
                skipped += n;
            }
        }
    }
    const u64 accounted = built + already + skipped;
    const u64 lost = list->entries.size() > accounted ? list->entries.size() - accounted : 0;
    if (lost != 0) {
        reasons[fmt::format("{} build process(es) ended without a result", failed)] += lost;
    }
    const u64 rescanned = ShaderStore::Rescan();
    const auto ms = static_cast<u64>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
    // the marker only for a complete build: a part that stopped early or a write that failed (a full disk) is built again at the next
    // start (the stages built are in the folder and are not translated again); entries a process stopped on stay left out (that file)
    const bool complete = failed == 0 && write_failed == 0 && lost == 0;
    if (complete) {
        std::ofstream f{ShaderList::BuiltMarker(store), std::ios::trunc};
        f << marker_line << "\n"
          << "entries " << list->entries.size() << "\nbuilt " << built << "\nalready " << already << "\nskipped " << skipped << "\nnot accounted " << lost
          << "\nwrite_failed " << write_failed << "\nfixed_dx10_clamp " << fixed_dx10_clamp << "\nfixed_depth_before_shader " << fixed_depth_before_shader
          << "\nrestarts " << restarts << "\nparts " << jobs << "\nparts_failed " << failed << "\nms " << ms << "\n";
        for (const auto& [reason, n] : reasons) {
            f << "left " << n << " " << reason << "\n";
        }
    }
    if (!complete) {
        std::printf("Shader prebuild (new engine): the build is not complete; no marker, the next start builds the rest\n");
    }
    for (const auto& [reason, n] : reasons) {
        std::printf("Shader prebuild (new engine): %llu list entries not built: %s\n", static_cast<unsigned long long>(n), reason.c_str());
    }
    std::printf("Shader prebuild (new engine): background build done in %llu ms: %zu entries, %llu built, %llu already in the store, %llu not built (%llu "
                "skipped, %llu not accounted for); fixed values set: dx10_clamp %llu, depth_before_shader %llu; %u build processes, %llu restarts, %llu stopped "
                "early, %llu writes failed; %llu new stage files read\n",
                static_cast<unsigned long long>(ms), list->entries.size(), static_cast<unsigned long long>(built), static_cast<unsigned long long>(already),
                static_cast<unsigned long long>(skipped + lost), static_cast<unsigned long long>(skipped), static_cast<unsigned long long>(lost),
                static_cast<unsigned long long>(fixed_dx10_clamp), static_cast<unsigned long long>(fixed_depth_before_shader), jobs,
                static_cast<unsigned long long>(restarts), static_cast<unsigned long long>(failed), static_cast<unsigned long long>(write_failed),
                static_cast<unsigned long long>(rescanned));
    if (failed != 0 || write_failed != 0 || lost != 0) {
        LOG_WARNING(Render_Vulkan, "vkb shader pre-build: {} build processes stopped early, {} writes failed, {} entries not accounted for (see the lines above)",
                    failed, write_failed, lost);
    }
    {
        std::scoped_lock lock{st.mutex};
        st.list = list_path.string();
        st.outcome = failed != 0 || lost != 0 ? "done, with build processes that stopped early" : "done";
        st.built = built;
        st.fixed_dx10_clamp = fixed_dx10_clamp;
        st.fixed_depth_before_shader = fixed_depth_before_shader;
        st.already = already;
        st.skipped = skipped + lost;
        st.write_failed = write_failed;
        st.restarts = restarts;
        st.parts = jobs;
        st.parts_failed = failed;
        st.accounted = accounted;
        st.ms = ms;
        st.rescanned = rescanned;
        st.reasons = std::move(reasons);
    }
    Finish();
    PipelineWarm::Start(); // the pipelines from the stored stages (native_vkb_pipeline_warm.h)
}

} // namespace

bool HookWanted() {
    static const bool on = [] {
        const char* vkb = std::getenv("TLG_VKB");
        return vkb != nullptr && vkb[0] == '1' && ShaderStore::OnDisk(); // TLG_VKB_SHADER_PREBUILD (not the session memo alone)
    }();
    return on;
}

void Start(const std::filesystem::path& store_folder) {
    static std::once_flag once;
    if (!HookWanted()) {
        return;
    }
    std::call_once(once, [&] {
        const auto list = ShaderList::Path();
        if (list.empty()) {
            std::printf("Shader prebuild (new engine): no shader list for this game (shader-list/shaders.tlgsl, the normal renderer's; TLG_VKB_SHADER_LIST)\n");
            std::scoped_lock lock{S().mutex};
            S().outcome = "no list";
            return;
        }
        const auto game = ShaderList::HostGameDir();
        if (game.empty()) {
            std::printf("Shader prebuild (new engine): the game folder is packed (.zar); the shader list is not read for it yet\n");
            std::scoped_lock lock{S().mutex};
            S().outcome = "packed game folder";
            return;
        }
        {
            std::scoped_lock lock{S().mutex};
            S().running = true;
            S().started = std::chrono::steady_clock::now();
            S().outcome = "running";
        }
        std::thread{[store_folder, list, game] { Run(store_folder, list, game); }}.detach();
    });
}

void StageLoad(const char* what, bool first) {
    if (!HookWanted()) {
        return;
    }
    State& st = S();
    const u64 before = ShaderStore::Rescan(); // the files the build wrote so far
    std::unique_lock lock{st.mutex};
    if (first || !st.running) {
        const bool running = st.running;
        lock.unlock();
        std::printf("Shader prebuild (new engine): stage load%s: %llu new stage files read%s\n", what, static_cast<unsigned long long>(before),
                    first && running ? " (first of the session: no wait, the build runs on)" : running ? "" : " (no build running)");
        return;
    }
    const long limit_s = EnvSeconds("TLG_VKB_SHADER_PREBUILD_WAIT_S", 120, 0, 3600);
    const auto t0 = std::chrono::steady_clock::now();
    std::printf("Shader prebuild (new engine): stage load%s waits for the background build (at most %ld s)\n", what, limit_s);
    st.done.wait_for(lock, std::chrono::seconds{limit_s}, [&] { return !st.running; });
    const bool gave_up = st.running;
    st.waits_given_up += gave_up;
    lock.unlock();
    const u64 after = ShaderStore::Rescan();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::printf("Shader prebuild (new engine): stage load waited %lld ms%s; %llu new stage files read\n", static_cast<long long>(ms),
                gave_up ? ", stopped waiting: the build is still running" : "", static_cast<unsigned long long>(before + after));
}

bool Running() {
    std::scoped_lock lock{S().mutex};
    return S().running;
}

bool Listed(u64 pgm_hash) {
    std::scoped_lock lock{S().mutex};
    return std::ranges::binary_search(S().listed, pgm_hash);
}

u32 WaitsGivenUp() {
    std::scoped_lock lock{S().mutex};
    return S().waits_given_up;
}

nlohmann::json Report() {
    State& st = S();
    std::scoped_lock lock{st.mutex};
    nlohmann::json left = nlohmann::json::object();
    for (const auto& [reason, n] : st.reasons) {
        left[reason] = n;
    }
    return {{"on", HookWanted()},       {"outcome", st.outcome}, {"list", st.list},        {"entries", st.entries},       {"built", st.built},
            {"already", st.already}, {"fixed_dx10_clamp", st.fixed_dx10_clamp}, {"fixed_depth_before_shader", st.fixed_depth_before_shader},    {"not_built", st.skipped}, {"not_built_by_reason", left}, {"write_failed", st.write_failed},
            {"restarts", st.restarts},  {"parts", st.parts},     {"parts_stopped_early", st.parts_failed}, {"ms", st.ms},
            {"files_read_after", st.rescanned}, {"running", st.running}, {"waits_given_up", st.waits_given_up},
            {"processes_without_job_object", unguarded.load()}};
}

} // namespace Core::Vkb::Prebuild
