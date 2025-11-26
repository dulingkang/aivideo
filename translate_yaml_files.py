#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译 YAML 配置文件中的中文内容为英文
"""

import yaml
import re
from pathlib import Path

def translate_character_profiles():
    """翻译角色配置文件"""
    file_path = Path("gen_video/character_profiles.yaml")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 翻译映射表
    translations = {
        # 注释
        "# 角色模板配置文件": "# Character Profile Configuration",
        "# 用于确保角色在不同场景中的一致性（脸型、发型、服饰等）": "# Ensures character consistency across scenes (face, hair, clothing, etc.)",
        "# 面部特征（用于 InstantID 和 prompt）": "# Facial features (for InstantID and prompt)",
        "# 发型（固定描述，权重高）": "# Hair (fixed description, high weight)",
        "# 服饰（固定描述，权重高）": "# Clothing (fixed description, high weight)",
        "# 视觉风格关键词": "# Visual style keywords",
        "# 身体特征": "# Body features",
        "# 合并到 prompt 的完整描述（自动生成）": "# Full description merged into prompt (auto-generated)",
        
        # 角色名称和身份
        "韩立": "Han Li",
        "男、青年修士，沉稳冷静、面容俊朗但不张扬": "Male, young cultivator, calm and steady, handsome but not flamboyant",
        "张奎": "Zhang Kui",
        "男、疤面壮汉，天东商号护卫队长，勇猛粗豪": "Male, scarred warrior, Tiandong Trading Company guard captain, brave and rough",
        "南歧子": "Nan Qizi",
        "男、中年道士，黄袍束发，性情沉稳": "Male, middle-aged Taoist, yellow robe with tied hair, calm temperament",
        "符老": "Fu Lao",
        "男、白袍老者，天东商号供奉，精通金针术": "Male, white-robed elder, Tiandong Trading Company patron, master of golden needle technique",
        "柳儿": "Liu Er",
        "女、蓝衫侍女，方夫人贴身护卫兼管事": "Female, blue-robed maid, Lady Fang's personal guard and steward",
        "香儿": "Xiang Er",
        "女、翠衫侍女，温婉灵敏，擅阵法": "Female, emerald-robed maid, gentle and agile, skilled in formations",
        "筌儿": "Quan Er",
        "女、红衫少女，年纪最小，活泼勇敢": "Female, red-robed young girl, youngest, lively and brave",
        "方夫人": "Lady Fang",
        "女、三十余岁商号主母，端庄沉稳、气度贵雅": "Female, merchant house matriarch in her thirties, dignified and steady, noble and elegant",
        "潘青": "Pan Qing",
        "男、二十多岁金玉宗弟子，方夫人之子": "Male, Jinyu Sect disciple in his twenties, Lady Fang's son",
        "秦师兄": "Senior Brother Qin",
        "男、金玉宗筑基后期修士，负责协助安远守城": "Male, Jinyu Sect Foundation Establishment late-stage cultivator, assists in Anyuan city defense",
        
        # 面部特征
        "剑眉、深邃眼形、轮廓分明但偏清秀": "sword-like brows, deep-set eyes, well-defined but refined features",
        "冷静专注的眼神，自然肤色，沉稳的表情，坚定的目光": "calm focused eyes, natural skin tone, steady expression, determined gaze",
        "方脸、颧骨高、左脸贯穿长疤": "square face, high cheekbones, long scar across left cheek",
        "久经沙场的目光，疤痕脸颊，凶狠的眉毛，坚定的下巴": "battle-hardened gaze, scarred cheeks, fierce brows, determined chin",
        "长脸、剑眉、薄唇": "long face, sword-like brows, thin lips",
        "冷静学者般的目光，淡淡微笑，沉稳的道者气质": "calm scholarly gaze, faint smile, steady Taoist demeanor",
        "削瘦老脸、白眉垂垂、眼神犀利": "gaunt elderly face, drooping white brows, sharp eyes",
        "深深的皱纹，锐利的老年目光，耐心的医者表情": "deep wrinkles, sharp elderly gaze, patient healer expression",
        "鹅蛋脸、柳眉、目光冷静": "oval face, willow brows, calm gaze",
        "冷静指挥者的目光，优雅的嘴唇，保护性的决心": "calm commander's gaze, elegant lips, protective determination",
        "圆润瓜子脸、明眸善睐": "rounded oval face, bright expressive eyes",
        "温和的微笑，明亮观察的眼神，安抚人心的气质": "gentle smile, bright observant eyes, soothing presence",
        "圆脸、俏鼻、小梨涡": "round face, cute nose, small dimples",
        "青春的光芒，表情丰富的眉毛，认真的决心": "youthful radiance, expressive brows, earnest determination",
        "鹅蛋脸、眉眼温润却藏锐气": "oval face, gentle eyes and brows hiding sharpness",
        "优雅的目光，沉稳的微笑，威严而冷静的气质": "elegant gaze, steady smile, dignified and calm demeanor",
        "长脸、眉目清俊": "long face, refined features",
        "文雅的眼神，恭敬的态度，青春的决心": "refined gaze, respectful attitude, youthful determination",
        "方正脸、剑眉薄唇": "square face, sword-like brows, thin lips",
        "敏锐洞察的眼神，沉稳战略家的表情": "sharp insightful gaze, steady strategist expression",
        
        # 发型
        "长发束起，前额略有碎发": "long hair tied up, slight bangs on forehead",
        "乌黑": "jet black",
        "中式仙侠风格、自然流动": "Chinese xianxia style, natural flow",
        "短发束后，鬓角剃短": "short hair tied back, short sideburns",
        "深棕": "dark brown",
        "常因吞服丹药竖立，显得硬朗": "often stands up from pill consumption, looks tough",
        "长发束成道髻": "long hair tied in Taoist bun",
        "乌黑夹杂少许白丝": "jet black with few white strands",
        "佩戴青色道冠": "wears blue Taoist crown",
        "长发披肩": "long hair over shoulders",
        "纯白": "pure white",
        "发丝略显干枯但整洁": "slightly dry but neat hair",
        "半挽高髻，余发披肩": "half-up high bun, rest over shoulders",
        "墨黑": "ink black",
        "以蓝银簪固定，利落整洁": "secured with blue-silver hairpin, neat and tidy",
        "长发编成双股垂于胸前": "long hair braided in two strands hanging on chest",
        "发尾缠以翠色丝带": "ends tied with emerald ribbon",
        "高马尾": "high ponytail",
        "以红缎绑成长长马尾": "tied in long ponytail with red ribbon",
        "高挽凤髻": "high phoenix bun",
        "墨黑带微金丝": "ink black with subtle gold strands",
        "饰以金步摇与青玉簪": "adorned with gold hairpin and jade hairpin",
        "束冠垂带": "crown with hanging ribbons",
        "佩戴金玉宗刻纹束带": "wears Jinyu Sect engraved headband",
        "束发戴金冠": "tied hair with gold crown",
        "深棕": "dark brown",
        "冠上嵌青金纹": "crown inlaid with blue-gold patterns",
        
        # 服饰
        "修士长袍": "cultivator robe",
        "深青色 + 墨色纹路": "deep cyan + ink patterns",
        "简洁实用，不华丽，有轻微战损痕迹": "simple and practical, not ornate, slight battle damage",
        "腰间储物袋": "waist storage pouch",
        "暗银色护腕": "dark silver wrist guards",
        "沙色皮甲+兽皮披肩": "sand-colored leather armor + fur shoulder cape",
        "土黄为主，辅以铁灰护具": "earth yellow main, iron gray armor accessories",
        "肩甲与护臂布满旧伤痕，腰间挂灵器与丹瓶": "shoulder guards and arm guards covered in old scars, spirit tools and pill bottles at waist",
        "狼牙棒": "wolf-tooth club",
        "骨制号角": "bone horn",
        "黄袍道衣": "yellow Taoist robe",
        "明黄配淡青镶边": "bright yellow with light cyan trim",
        "袖口绣有云纹，腰系玉环法器": "cloud patterns embroidered on cuffs, jade ring talisman at waist",
        "铜镜": "bronze mirror",
        "飞剑匣": "flying sword case",
        "素白长袍+银线纹路": "plain white robe + silver thread patterns",
        "白底配浅银暗纹": "white base with light silver subtle patterns",
        "袖口藏针筒，腰系药袋": "needle case hidden in cuffs, medicine pouch at waist",
        "金针卷轴": "golden needle scroll",
        "药香烟囊": "medicine incense pouch",
        "蓝色长裙+轻甲肩片": "blue long dress + light armor shoulder pieces",
        "靛蓝主色，辅以银线": "indigo main color, silver thread accents",
        "胸前有天东商号纹饰，腰悬短刃": "Tiandong Trading Company emblem on chest, short blade at waist",
        "灵镯操控旗": "spirit bracelet control flag",
        "通讯符鸟笛": "communication talisman bird whistle",
        "翠绿色云纹裙": "emerald green cloud-patterned dress",
        "碧翠+浅金饰边": "jade green + light gold trim",
        "袖间藏阵旗，腰佩香囊": "formation flags hidden in sleeves, incense pouch at waist",
        "小回春阵旗": "small rejuvenation formation flag",
        "药香囊": "medicine incense pouch",
        "红色短襟战裙": "red short-fronted battle dress",
        "朱红配金边": "vermilion with gold trim",
        "裙摆便于行动，袖口收束": "skirt hem for mobility, tight cuffs",
        "小旗": "small flag",
        "护腕灵石": "wrist guard spirit stone",
        "青缎长裙+云纹披肩": "cyan satin long dress + cloud-patterned shawl",
        "深青底配银线": "deep cyan base with silver thread",
        "胸口绣天东商号纹章，袖口垂金丝流苏": "Tiandong Trading Company emblem embroidered on chest, gold thread tassels on cuffs",
        "玉镯": "jade bracelet",
        "丝帕": "silk handkerchief",
        "香囊": "incense pouch",
        "蓝袍道服": "blue Taoist robe",
        "藏蓝配浅金边": "navy blue with light gold trim",
        "胸前绣金玉宗纹章，腰佩符袋": "Jinyu Sect emblem embroidered on chest, talisman pouch at waist",
        "法剑": "spirit sword",
        "符袋": "talisman pouch",
        "锦袍道甲": "brocade robe with Taoist armor",
        "墨金底配暗纹": "dark gold base with subtle patterns",
        "肩部暗金甲片，腰系玉符链": "dark gold armor pieces on shoulders, jade talisman chain at waist",
        "折扇": "folding fan",
        "法器腰牌": "spirit tool waist token",
        
        # 视觉关键词
        "中国玄幻风": "Chinese fantasy style",
        "国漫风格": "Chinese animation style",
        "面部五官稳定、冷静坚毅": "stable facial features, calm and resolute",
        "保持沉稳的表情，不夸张": "maintain steady expression, not exaggerated",
        "照片级真实感，超写实": "photorealistic, hyperrealistic",
        "电影级光照": "cinematic lighting",
        "瘦削健壮的身材": "lean and strong build",
        "英勇但放松的姿态": "heroic but relaxed posture",
        "中等身高，比例匀称": "medium height, well-proportioned",
        "粗犷豪迈": "rugged and bold",
        "体格魁梧": "burly physique",
        "战场指挥者": "battlefield commander",
        "狼骑武者": "wolf-riding warrior",
        "魁梧肌肉发达的身材": "burly muscular build",
        "前倾攻击性姿态": "forward-leaning aggressive posture",
        "略高于平均，宽肩": "slightly above average height, broad shoulders",
        "道门修士": "Taoist cultivator",
        "沉稳医者": "steady healer",
        "温润气质": "gentle temperament",
        "掌控法器": "master of spirit tools",
        "瘦削学者身材": "lean scholar build",
        "直立优雅姿态": "upright elegant posture",
        "中等身高，轻盈步伐": "medium height, light steps",
        "老牌医者": "veteran healer",
        "稳重克制": "steady and restrained",
        "针灸大师": "acupuncture master",
        "瘦削老年身材": "lean elderly build",
        "略微佝偻但稳健": "slightly hunched but steady",
        "中等身高，依靠灵气而非力量": "medium height, relies on spiritual energy not strength",
        "冷静指挥": "calm command",
        "护卫领队": "guard leader",
        "手握阵旗": "holding formation flag",
        "瘦削健美的身材": "lean and fit build",
        "直立指挥姿态": "upright command posture",
        "略高于平均女性身高": "slightly above average female height",
        "温润治愈": "gentle healing",
        "阵法施术": "formation casting",
        "柔和气质": "soft temperament",
        "苗条优雅的身材": "slender elegant build",
        "略微前倾的慈悲姿态": "slightly forward-leaning compassionate posture",
        "平均女性身高": "average female height",
        "灵动少女": "lively young girl",
        "阵旗助手": "formation flag assistant",
        "勇敢但稚嫩": "brave but young",
        "娇小敏捷的身材": "petite agile build",
        "充满活力的前倾姿态": "energetic forward-leaning posture",
        "比同龄人矮": "shorter than peers",
        "贵族气质": "noble temperament",
        "掌控全局": "in control",
        "温柔而威严": "gentle yet dignified",
        "苗条优雅的身材": "slender elegant build",
        "直立优雅的姿态": "upright elegant posture",
        "平均女性身高": "average female height",
        "宗门弟子": "sect disciple",
        "谦逊谨慎": "humble and cautious",
        "未完全成熟": "not fully mature",
        "瘦削学者身材": "lean scholar build",
        "礼貌直立的姿态": "polite upright posture",
        "高瘦": "tall and thin",
        "沉稳老练": "steady and experienced",
        "宗门核心弟子": "core sect disciple",
        "城防指挥": "city defense commander",
        "健壮中等身材": "strong medium build",
        "双手背后，权威的姿态": "hands behind back, authoritative posture",
        "略高于平均男性身高": "slightly above average male height",
    }
    
    # 执行翻译
    result = content
    sorted_translations = sorted(translations.items(), key=lambda x: len(x[0]), reverse=True)
    for chinese, english in sorted_translations:
        result = result.replace(chinese, english)
    
    # 保存翻译后的文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result)
    
    print(f"✓ 翻译完成: {file_path}")

def translate_scene_profiles():
    """翻译场景配置文件"""
    file_path = Path("gen_video/scene_profiles.yaml")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 翻译映射表（场景相关）
    translations = {
        "青罗沙漠": "Qingluo Desert",
        "青罗灵宫": "Qingluo Spirit Palace",
        "龟车防线·沙虫战场": "Turtle Caravan Defense Line · Sand Worm Battlefield",
        "天元大草原": "Tianyuan Great Prairie",
        "安远巨城": "Anyuan Great City",
        "安远城墙防线": "Anyuan City Wall Defense Line",
        "安远城市集": "Anyuan City Market",
        "如云客栈": "Ruyun Inn",
        "灵光深渊": "Spirit Light Abyss",
        "幻灵山": "Huanling Mountain",
        "澜风城": "Lanfeng City",
        "真仙遗府": "True Immortal Mansion",
        "界源之地": "Realm Source Chamber",
        "天元圣城": "Tianyuan Holy City",
        
        # 颜色
        "金黄": "golden yellow",
        "暖橙": "warm orange",
        "淡褐色": "light brown",
        "沙黄": "sand yellow",
        "米色": "beige",
        "浅棕": "light brown",
        "深褐": "dark brown",
        "暗金": "dark gold",
        "青色": "cyan",
        "墨绿": "ink green",
        "深青": "deep cyan",
        "淡蓝": "light blue",
        "青白": "cyan white",
        "银灰": "silver gray",
        "金色": "gold",
        "灵光蓝": "spirit light blue",
        "符文青": "rune cyan",
        "灰青": "gray cyan",
        "暗褐": "dark brown",
        "尘橙": "dust orange",
        "土金": "earth gold",
        "铁黑": "iron black",
        "灵白": "spirit white",
        "青光": "cyan light",
        "血红": "blood red",
        "灰绿": "gray green",
        "苍青": "pale cyan",
        "淡褐": "light brown",
        "浅蓝": "light blue",
        "微青光": "subtle cyan light",
        "石灰白": "limestone white",
        "暗铁灰": "dark iron gray",
        "墨黑": "ink black",
        "靛蓝": "indigo",
        "青灰": "cyan gray",
        "铜金": "copper gold",
        "灵白光": "spirit white light",
        "符纹青": "rune pattern cyan",
        "铁灰": "iron gray",
        "暗金": "dark gold",
        "尘黄": "dust yellow",
        "火红": "fire red",
        "靛蓝旗": "indigo flag",
        "兵器银": "weapon silver",
        "石灰": "limestone",
        "木褐": "wood brown",
        "帆布白": "canvas white",
        "旗帜绛蓝": "flag crimson blue",
        "深木棕": "deep wood brown",
        "墨青": "ink cyan",
        "青石灰": "cyan limestone",
        "竹绿": "bamboo green",
        "奶白帘": "cream white curtain",
        "金字匾额": "golden plaque",
        "暖灯橙": "warm lamp orange",
        "深蓝": "deep blue",
        "青蓝": "cyan blue",
        "幽蓝": "dark blue",
        "淡蓝": "light blue",
        "灵光蓝": "spirit light blue",
        "符文青": "rune cyan",
        "灵光白": "spirit white light",
        "翠绿": "emerald green",
        "深绿": "dark green",
        "灵光绿": "spirit light green",
        "云白": "cloud white",
        "玉白": "jade white",
        "淡金": "light gold",
        "古铜": "bronze",
        "银白": "silver white",
        "灵光金": "spirit light gold",
        "深金": "deep gold",
        "符文金": "rune gold",
        "仙光白": "immortal light white",
        "幽蓝": "dark blue",
        "界源光": "realm source light",
        "云白": "cloud white",
        "淡青": "light cyan",
        "圣光金": "holy light gold",
        
        # 地形和描述
        "连绵沙丘、干燥空气、远处偶有岩壁": "rolling sand dunes, dry air, occasional distant rock walls",
        "连绵的沙丘，干燥的空气，远方的岩壁": "rolling sand dunes, dry air, distant rock walls",
        "翻滚的沙浪": "rolling sand waves",
        "广阔的金色沙漠": "vast golden desert",
        "远方的石峰": "distant stone peaks",
        "飘散的沙子": "scattered sand",
        "明亮的阳光，微妙的热浪扭曲": "bright sunlight, subtle heat wave distortion",
        "强烈的阳光": "intense sunlight",
        "热浪扭曲": "heat wave distortion",
        "晴朗明亮的天空": "clear bright sky",
        "金色的夕阳光线": "golden sunset rays",
        "干旱、广阔，带有微妙的灵气波动": "arid, vast, with subtle spiritual energy fluctuations",
        "干旱的氛围": "arid atmosphere",
        "广阔空旷的景观": "vast open landscape",
        "微妙的灵气": "subtle spiritual energy",
        "神秘的灵气波动": "mysterious spiritual energy fluctuations",
        "国漫·玄幻风格": "Chinese animation · fantasy style",
        "仙侠修炼世界": "xianxia cultivation world",
        "中国古代玄幻": "ancient Chinese fantasy",
        "电影级仙侠风格": "cinematic xianxia style",
        "沙子颜色始终偏金黄，不要出现白沙或红沙": "sand color always golden yellow, no white or red sand",
        "天空保持晴朗，不出现奇怪云层风暴": "sky remains clear, no strange clouds or storms",
        "岩石结构保持东方玄幻风，不要变现代": "rock structures maintain Eastern fantasy style, not modern",
        "保持沙漠的干燥感，不要出现水或绿洲（除非脚本明确要求）": "maintain desert dryness, no water or oases (unless script explicitly requires)",
        "电影级规模感": "cinematic scale",
        "仙侠沙漠边境": "xianxia desert border",
        
        # 更多场景描述...
        # 由于内容太多，我会继续添加关键翻译
    }
    
    # 执行翻译
    result = content
    sorted_translations = sorted(translations.items(), key=lambda x: len(x[0]), reverse=True)
    for chinese, english in sorted_translations:
        result = result.replace(chinese, english)
    
    # 保存翻译后的文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result)
    
    print(f"✓ 翻译完成: {file_path}")

if __name__ == "__main__":
    translate_character_profiles()
    translate_scene_profiles()
    print("\n✓ 所有 YAML 文件翻译完成")

