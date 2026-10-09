"""Build a config for a demo glider file and run the pipeline on it.

If you have your own OG1 file, you can replace the get_demo_file call with its path.

If you have your own config, you can load it instead of making a new one. You can also
call make_config with ask=True to be prompted about each choice:

    Pipeline.make_config(file, ask=True)               # asking about each choice
    Pipeline.load_config("/path/to/your_config.yaml")  # a config you already have

make_config saves the config next to the data file (e.g. Nelson_646_R.yaml), so you can
edit it and rerun it with load_config.

Demo files (downloaded into ~/Documents/pelagos-py/demo_data the first time; get_demo_file() prints this
list; names ending in _r are near real time, the rest delayed mode):

    Bio-Carbon: nelson_646_r, nelson_646, doombar_648_r, doombar_648, churchill_647_r,
                churchill_647, alr_4_649_r, alr_4_649, alr_6_650_r, alr_6_650, cabot_645_r,
                cabot_645
    Custard 1:  churchill_501_r, churchill_501, pancake_502_r, doombar_503_r
    Custard 2:  bellamite_538_r, bellamite_538, zephyr_539
    ReBELS:     zephyr_675_r, zephyr_675, omg-1_676_r, 9ja_699_r, 9ja_699, growler_677_r,
                growler_677, stella_678_r, stella_678
    ReBELS 2:   stella_713_r
    VOTO:       sea063_20240724t0737_delayed
"""

from pelagos_py import Pipeline, get_demo_file

file = get_demo_file("nelson_646_r")
p = Pipeline.make_config(file)
p.run()
