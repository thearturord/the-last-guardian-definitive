/* TLG_PAD_MAP / TLG_KEY_MAP parsing (src/runtime_pad.c). */
#define _GNU_SOURCE
#include <assert.h>
#include "../src/runtime_pad.c"

void tlggpu_pad_game_buttons(uint32_t buttons) { (void)buttons; }
void tlggpu_pad_kind(int xbox) { (void)xbox; }
int tlggpu_overlay_captures_input(void) { return 0; }
void tlggpu_mouse_take(TlgGpuMouse *m) { memset(m,0,sizeof(*m)); }
void tlggpu_input_settings(TlgGpuInputSettings *s) { memset(s,0,sizeof(*s)); }
uintptr_t runtime_lookup(const RuntimeExport *table, size_t count, const char *name) {
    (void)table; (void)count; (void)name;
    return 0;
}

/* PS4 buttons a host source presses in `map`. */
static uint32_t pressed_by(const ButtonMap *map, int source) {
    uint32_t ps=0;
    for (size_t i=0;i<map->count;++i) if (map->entries[i].source==source) ps|=map->entries[i].ps;
    return ps;
}

int main(void) {
    ButtonMap keys=key_map;
    assert(apply_button_map(&keys,NULL,true)==0 && apply_button_map(&keys,"",true)==0);
    assert(keys.count==key_map.count);

    /* Swap: Q presses Cross, Space presses Triangle; Space no longer presses Cross. */
    assert(apply_button_map(&keys," cross = Q , triangle=Space",true)==2);
    assert(pressed_by(&keys,SDL_SCANCODE_Q)==BTN_CROSS);
    assert(pressed_by(&keys,SDL_SCANCODE_SPACE)==BTN_TRIANGLE);
    assert(pressed_by(&keys,SDL_SCANCODE_E)==BTN_SQUARE);

    /* Names with spaces, any case; the last entry for a button wins. */
    ButtonMap again=key_map;
    assert(apply_button_map(&again,"CIRCLE=left shift,circle=Left Ctrl",true)==2);
    assert(pressed_by(&again,SDL_SCANCODE_LCTRL)==BTN_CIRCLE);
    assert(pressed_by(&again,SDL_SCANCODE_LSHIFT)==0);

    /* Bad entries are skipped and leave the defaults alone. */
    ButtonMap bad=key_map;
    assert(apply_button_map(&bad,"bogus=Q,cross=NotAKey,circle,=Q,square=,,",true)==0);
    assert(bad.count==key_map.count && pressed_by(&bad,SDL_SCANCODE_SPACE)==BTN_CROSS);

    /* Gamepad: Xbox layout swap; 'touchpad' is the left click, Back stays the right click and
     * the real touchpad click keeps its fingers. */
    ButtonMap pad=pad_map;
    assert(apply_button_map(&pad,"cross=b,circle=a,touchpad=misc1",false)==3);
    assert(pressed_by(&pad,SDL_GAMEPAD_BUTTON_EAST)==BTN_CROSS);
    assert(pressed_by(&pad,SDL_GAMEPAD_BUTTON_SOUTH)==BTN_CIRCLE);
    assert(pressed_by(&pad,SDL_GAMEPAD_BUTTON_MISC1)==CLICK_LEFT);
    assert(pressed_by(&pad,SDL_GAMEPAD_BUTTON_BACK)==CLICK_RIGHT);
    assert(pressed_by(&pad,SDL_GAMEPAD_BUTTON_TOUCHPAD)==BTN_TOUCHPAD);
    assert(apply_button_map(&pad,"touchpad_right=guide,touchpad_left=paddle1",false)==2);
    assert(pressed_by(&pad,SDL_GAMEPAD_BUTTON_BACK)==0 && pressed_by(&pad,SDL_GAMEPAD_BUTTON_GUIDE)==CLICK_RIGHT);
    assert(pressed_by(&pad,SDL_GAMEPAD_BUTTON_MISC1)==0 && pressed_by(&pad,SDL_GAMEPAD_BUTTON_RIGHT_PADDLE1)==CLICK_LEFT);
    assert(apply_button_map(&pad,"cross=Space",false)==0); /* a key name is no gamepad button */
    /* Keyboard: each side has a key of its own. */
    assert(pressed_by(&key_map,SDL_SCANCODE_TAB)==CLICK_LEFT && pressed_by(&key_map,SDL_SCANCODE_BACKSPACE)==CLICK_RIGHT);
    ButtonMap sides=key_map;
    assert(apply_button_map(&sides,"touchpad_right=Delete",true)==1);
    assert(pressed_by(&sides,SDL_SCANCODE_BACKSPACE)==0 && pressed_by(&sides,SDL_SCANCODE_DELETE)==CLICK_RIGHT);
    /* Mouse defaults: left Triangle, right R1, middle Square; mouse names in the keyboard map. */
    assert(pressed_by(&key_map,MOUSE_LEFT)==BTN_TRIANGLE && pressed_by(&key_map,MOUSE_RIGHT)==BTN_R1);
    assert(pressed_by(&key_map,MOUSE_MIDDLE)==BTN_SQUARE && pressed_by(&key_map,WHEEL_UP)==0);
    ButtonMap mouse=key_map;
    assert(apply_button_map(&mouse,"cross=mouse x1,up=Wheel Up,down=Wheel Down",true)==3);
    assert(pressed_by(&mouse,MOUSE_X1)==BTN_CROSS && pressed_by(&mouse,SDL_SCANCODE_SPACE)==0);
    assert(pressed_by(&mouse,WHEEL_UP)==BTN_UP && pressed_by(&mouse,SDL_SCANCODE_I)==0);
    assert(pressed_by(&mouse,WHEEL_DOWN)==BTN_DOWN);
    assert(apply_button_map(&mouse,"cross=Mouse X1",false)==0); /* not a gamepad button */
    /* '|': several sources; "none": no source; an unknown part is skipped. */
    ButtonMap multi=key_map;
    assert(apply_button_map(&multi,"triangle=Q|Mouse Left|Bogus,r1=none,l1=3 | Mouse Right",true)==3);
    assert(pressed_by(&multi,SDL_SCANCODE_Q)==BTN_TRIANGLE && pressed_by(&multi,MOUSE_LEFT)==BTN_TRIANGLE);
    assert(pressed_by(&multi,SDL_SCANCODE_3)==BTN_L1 && pressed_by(&multi,MOUSE_RIGHT)==BTN_L1);
    assert(pressed_by(&multi,SDL_SCANCODE_1)==0);
    for (size_t i=0;i<multi.count;++i) assert(multi.entries[i].ps!=BTN_R1);
    ButtonMap gp=pad_map;
    assert(apply_button_map(&gp,"cross=a|b",false)==1);
    assert(pressed_by(&gp,SDL_GAMEPAD_BUTTON_SOUTH)==BTN_CROSS && pressed_by(&gp,SDL_GAMEPAD_BUTTON_EAST)==(BTN_CROSS|BTN_CIRCLE));
    /* Issue #104: movement, camera and motion keys are in the keyboard map too. ESDF moves, W
     * is free for a button and no longer moves; the gamepad map does not take these names. */
    assert(pressed_by(&key_map,SDL_SCANCODE_W)==KEY_MOVE_UP && pressed_by(&key_map,SDL_SCANCODE_LEFT)==KEY_CAM_LEFT);
    assert(pressed_by(&key_map,SDL_SCANCODE_T)==KEY_TILT_RIGHT && pressed_by(&key_map,SDL_SCANCODE_Y)==KEY_SHAKE);
    ButtonMap move=key_map;
    assert(apply_button_map(&move,"move_up=E,move_down=D,move_left=S,move_right=F,square=W,r2=A,camera_left=Keypad 4,shake=none",true)==8);
    assert(pressed_by(&move,SDL_SCANCODE_E)==KEY_MOVE_UP && pressed_by(&move,SDL_SCANCODE_D)==KEY_MOVE_DOWN);
    assert(pressed_by(&move,SDL_SCANCODE_S)==KEY_MOVE_LEFT && pressed_by(&move,SDL_SCANCODE_F)==KEY_MOVE_RIGHT);
    assert(pressed_by(&move,SDL_SCANCODE_W)==BTN_SQUARE && pressed_by(&move,SDL_SCANCODE_A)==BTN_R2);
    assert(pressed_by(&move,SDL_SCANCODE_KP_4)==KEY_CAM_LEFT && pressed_by(&move,SDL_SCANCODE_LEFT)==0);
    assert(pressed_by(&move,SDL_SCANCODE_Y)==0 && pressed_by(&move,SDL_SCANCODE_RIGHT)==KEY_CAM_RIGHT);
    ButtonMap gp2=pad_map;
    assert(apply_button_map(&gp2,"move_up=a",false)==0 && gp2.count==pad_map.count);
    /* What is held: the actions stay apart from the PS4 buttons. */
    static bool held[SDL_SCANCODE_COUNT];
    const ButtonMap saved=key_map;
    key_map=move;
    held[SDL_SCANCODE_E]=held[SDL_SCANCODE_W]=true;
    assert(key_map_pressed(held,0)==(KEY_MOVE_UP|BTN_SQUARE));
    assert(key_map_pressed(held,1u<<(MOUSE_LEFT-MOUSE_SOURCE))==(KEY_MOVE_UP|BTN_SQUARE|BTN_TRIANGLE));
    assert(key_map_pressed(NULL,0)==0);
    key_map=saved;
    assert((KEY_ACTIONS & (BTN_TOUCHPAD|CLICK_LEFT|CLICK_RIGHT|0xffff))==0);
    puts("PASS: pad and key maps");
}
