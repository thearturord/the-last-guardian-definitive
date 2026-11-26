# Pad patterns for shot.py

`SHOT_PAD_PATTERN=pads/move-a.txt` makes `shot.py` play a timed pad pattern after the load (after its fixed
8 s wait, before the photo steps); the play time is at least the pattern's length (`SHOT_PLAY_S` can add to it).
Test tool only: it writes the game's `TLG_PAD_FILE` (src/runtime_pad.c), the same input a scripted run always
had; nothing in the game changes.

Format: one step per line, `SECONDS TOKENS...`, times rising. The tokens of a line hold from its time until the
next line; a line with no tokens lets go of everything; the last line ends the pattern. Tokens are those of
`TLG_PAD_FILE`: `lx= ly= rx= ry=` (0..255, 128 = middle, `ly=0` = stick forward) and button names (`cross`,
`triangle`, `r1`, ...). `#` starts a comment.

The game log gets `Lua: shot pad pattern start` and `... end`, and the game's own `Runtime: pad file:` line for
every step it read, so counters can be read from a fixed moment after the load.

- `move-a.txt`: shader list recording (camera circle right, walk forward/right/back/left, about 45 s).
- `move-b.txt`: the check run: similar but not the same (circle left, other walk order, two jumps).
