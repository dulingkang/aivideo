#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译剩余的英文 action 和 motion 字段
"""

import json
import re
from pathlib import Path

# 扩展的 action 映射表
extended_action_map = {
    "meditate": "打坐",
    "ballista_fire": "发射弩箭",
    "halt_at_mound": "在土丘处停下",
    "explain_purpose": "解释目的",
    "examine_body": "检查身体",
    "diagnose": "诊断",
    "arrange_cushion": "安排坐垫",
    "second_acupuncture": "第二次针灸",
    "move_arms": "移动手臂",
    "show_amazement": "表现惊讶",
    "receive_talismans": "接收符箓",
    "chat_with_girls": "与女孩聊天",
    "carriage_stops": "马车停下",
    "decide_release_beast": "决定释放妖兽",
    "form_defense": "形成防御",
    "assemble_troops": "集结部队",
    "consume_pill_transform": "服用丹药变身",
    "charge_into_dust": "冲入尘土",
    "volley_fire": "齐射",
    "reveal_monsters": "揭示怪物",
    "sustain_losses": "承受损失",
    "internal_monologue": "内心独白",
    "decide_spell": "决定法术",
    "assemble_array": "组装阵法",
    "activate_array": "激活阵法",
    "analyze_magic": "分析法术",
    "monster_emergence": "怪物出现",
    "stone_rescue": "石头救援",
    "hanli_hit": "韩立击中",
    "check_injury": "检查伤势",
    # 连写单词
    "announcevictory": "宣布胜利",
    "sharebackground": "分享背景",
    "remarkmystery": "评论神秘",
    "airpatrol": "空中巡逻",
    "discussbeasttide": "讨论兽潮",
    "ponderorders": "思考命令",
    "firstsightcity": "初次看到城市",
    "citydefenseready": "城市防御就绪",
    "presenttoken": "出示令牌",
    "magicalscan": "法术扫描",
    "inspectweapon": "检查武器",
    "buymaterial": "购买材料",
    "paywithspiritstone": "用灵石支付",
    "readinncorner": "在客栈角落阅读",
    "visualizemap": "可视化地图",
    "ponderarray": "思考阵法",
    "studybeasttide": "研究兽潮",
    "considerheavenlytribulation": "考虑天劫",
    "decidetraining": "决定训练",
    "destroybooks": "销毁书籍",
    # 更多连写单词
    "commissionweapon": "委托武器",
    "hirecart": "雇佣马车",
    "tipservant": "给小费给仆人",
    "throwguard": "扔给守卫",
    "gainadmission": "获得准入",
    "receiveassignment": "接收任务",
    "plansecret": "计划秘密",
    "plotstrategy": "策划策略",
    "sealpact": "封印契约",
    "showstrength": "展示实力",
    "studymanual": "研究手册",
    "calmtroops": "安抚部队",
    "citylockdown": "城市封锁",
    "receivereport": "接收报告",
    "discusssilverwolf": "讨论银狼",
    "evaluatethreat": "评估威胁",
    "receiveorders": "接收命令",
    "marchout": "出发",
    "assembly": "集结",
    "distributeweapons": "分发武器",
    # 更多连写单词
    "aerialduel": "空中对决",
    "alarmsignal": "警报信号",
    "ambushtrap": "伏击陷阱",
    "assembleelite": "集结精英",
    "blackphoenixappears": "黑凤凰出现",
    "braceassault": "准备攻击",
    "childclings": "孩子紧抱",
    "confrontpuppet": "面对傀儡",
    "countertail": "反击尾部",
    "cultivatorslaunch": "修士发射",
    "detecthanli": "探测韩立",
    "emergealive": "活着出现",
    "explainmission": "解释任务",
    "expressgratitude": "表达感激",
    "extendinvitation": "发出邀请",
    "familyreunion": "家庭团聚",
    "farewell": "告别",
    "groundswallowsrider": "地面吞没骑手",
    "introducechild": "介绍孩子",
    "keepalert": "保持警觉",
    "offerreward": "提供奖励",
    "ominousreveal": "不祥揭示",
    "preparemiexianbeads": "准备灭仙珠",
    "pythonlaunch": "蟒蛇发射",
    "questionhanli": "询问韩立",
    "reactinformation": "反应信息",
    "revealself": "揭示自己",
    "ridetofront": "骑到前线",
    "slaughterfollowers": "屠杀追随者",
    "snakesbite": "蛇咬",
    "speculate": "推测",
    "subtlecomment": "微妙评论",
    "tailing": "跟踪",
    "tracktomound": "追踪到土丘",
    "wallcollapse": "墙倒塌",
    "withdraw": "撤退",
    "wolvesretreat": "狼群撤退",
}

# 扩展的 motion 映射表
extended_motion_map = {
    "slow push": "缓慢推进",
    "slow zoom": "缓慢变焦",
    "slow pan": "缓慢平移",
    "slight dolly": "轻微移动",
    "fast pan": "快速平移",
    "fast tracking": "快速跟踪",
    "medium shot": "中景",
    "tracking shot": "跟踪镜头",
    "panning": "平移",
    "handheld": "手持",
    "slow orbit": "缓慢环绕",
    "tracking": "跟踪",
    "slow dolly": "缓慢移动",
    "gentle pan": "轻柔平移",
    "slight tilt": "轻微倾斜",
    "slow tilt": "缓慢倾斜",
    "crane shot": "升降镜头",
    "rapid pan": "快速平移",
    "tracking bolt": "跟踪闪电",
    "rapid sweep": "快速扫过",
    "fast orbit": "快速环绕",
    "freeze-shot": "冻结镜头",
    "close-up": "特写",
    "close-up of door": "门特写",
    "close-up of face": "面部特写",
    "medium shot of entry": "入口中景",
    "medium shot of explanation": "解释中景",
    "close-up of ring and mirror": "戒指和镜子特写",
    "medium shot of examination": "检查中景",
    "medium shot of diagnosis": "诊断中景",
    # 连写单词
    "dronespinrevealingfullformation": "无人机旋转展现完整阵法",
    "push-inaswormsleap": "推进当虫跃起",
    "sweepingoverheadtrackshowingchaos": "扫过上方跟踪展现混乱",
    "wavefronttravelingfromcentertoedges": "波前从中心向边缘传播",
    "slow-motionfollowthroughtocrashingintocrate": "慢动作跟随到撞入箱子",
    "montagedissolvesfromgravestorestingHanLi": "蒙太奇从坟墓淡入到休息的韩立",
    "arclightsacrossline": "弧光划过线条",
    "centeredforwardmove": "中心向前移动",
    "macroslide": "宏滑动",
    "top-downdrift": "自上而下漂移",
    "freeze-then-slow": "冻结然后慢动作",
    "follow": "跟随",
    "follow-through": "跟随通过",
    "push-in": "推进",
    "aerialchase": "空中追逐",
    "subtlesway": "轻微摇摆",
    "slow-motiondodge": "慢动作闪避",
    "lowanglepush-in": "低角度推进",
    "lowanglereveal": "低角度揭示",
}

# 通用翻译函数：将英文单词翻译成中文
def translate_english_word(word: str) -> str:
    """将单个英文单词翻译成中文"""
    word_lower = word.lower().strip()
    
    # 常见动作词汇映射
    word_map = {
        # 动作相关
        "announce": "宣布", "victory": "胜利", "share": "分享", "background": "背景",
        "remark": "评论", "mystery": "神秘", "patrol": "巡逻", "air": "空中",
        "discuss": "讨论", "beast": "妖兽", "tide": "潮", "ponder": "思考",
        "orders": "命令", "sight": "看到", "city": "城市", "defense": "防御",
        "ready": "就绪", "present": "出示", "token": "令牌", "magical": "法术",
        "scan": "扫描", "inspect": "检查", "weapon": "武器", "buy": "购买",
        "material": "材料", "pay": "支付", "spirit": "灵", "stone": "石",
        "read": "阅读", "inn": "客栈", "corner": "角落", "visualize": "可视化",
        "map": "地图", "array": "阵法", "study": "研究", "consider": "考虑",
        "heavenly": "天", "tribulation": "劫", "decide": "决定", "training": "训练",
        "destroy": "销毁", "books": "书籍",
        "commission": "委托", "weapon": "武器", "hire": "雇佣", "cart": "马车",
        "tip": "小费", "servant": "仆人", "throw": "扔", "guard": "守卫",
        "gain": "获得", "admission": "准入", "receive": "接收", "assignment": "任务",
        "plan": "计划", "secret": "秘密", "plot": "策划", "strategy": "策略",
        "seal": "封印", "pact": "契约", "show": "展示", "strength": "实力",
        "study": "研究", "manual": "手册", "calm": "安抚", "troops": "部队",
        "city": "城市", "lockdown": "封锁", "report": "报告", "discuss": "讨论",
        "silver": "银", "wolf": "狼", "evaluate": "评估", "threat": "威胁",
        "orders": "命令", "march": "出发", "out": "出", "assembly": "集结",
        "distribute": "分发", "weapons": "武器",
        "aerial": "空中", "duel": "对决", "alarm": "警报", "signal": "信号",
        "ambush": "伏击", "trap": "陷阱", "elite": "精英", "black": "黑",
        "phoenix": "凤凰", "appears": "出现", "brace": "准备", "assault": "攻击",
        "child": "孩子", "clings": "紧抱", "confront": "面对", "puppet": "傀儡",
        "counter": "反击", "tail": "尾部", "cultivators": "修士", "launch": "发射",
        "detect": "探测", "emerge": "出现", "alive": "活着", "explain": "解释",
        "mission": "任务", "express": "表达", "gratitude": "感激", "extend": "发出",
        "invitation": "邀请", "family": "家庭", "reunion": "团聚", "farewell": "告别",
        "ground": "地面", "swallows": "吞没", "rider": "骑手", "introduce": "介绍",
        "keep": "保持", "alert": "警觉", "offer": "提供", "reward": "奖励",
        "ominous": "不祥", "reveal": "揭示", "prepare": "准备", "miexian": "灭仙",
        "beads": "珠", "python": "蟒蛇", "question": "询问", "react": "反应",
        "information": "信息", "self": "自己", "ride": "骑", "to": "到", "front": "前线",
        "slaughter": "屠杀", "followers": "追随者", "snakes": "蛇", "bite": "咬",
        "speculate": "推测", "subtle": "微妙", "comment": "评论", "tailing": "跟踪",
        "track": "追踪", "mound": "土丘", "wall": "墙", "collapse": "倒塌",
        "withdraw": "撤退", "wolves": "狼群", "retreat": "撤退",
        # 镜头相关
        "slow": "缓慢", "fast": "快速", "rapid": "快速",
        "slight": "轻微", "gentle": "轻柔",
        "push": "推进", "zoom": "变焦", "pan": "平移",
        "dolly": "移动", "tracking": "跟踪", "shot": "镜头",
        "orbit": "环绕", "tilt": "倾斜", "sweep": "扫过",
        "close": "特写", "up": "向上", "medium": "中景", "wide": "全景",
        "of": "", "the": "", "a": "", "an": "",
        "handheld": "手持", "crane": "升降", "freeze": "冻结",
        "bolt": "闪电", "entry": "入口", "explanation": "解释",
        "examination": "检查", "diagnosis": "诊断",
        "door": "门", "face": "面部", "ring": "戒指", "mirror": "镜子",
        "drone": "无人机", "spin": "旋转", "reveal": "展现", "full": "完整",
        "formation": "阵法", "worm": "虫", "leap": "跃起", "overhead": "上方",
        "chaos": "混乱", "wave": "波", "front": "前", "travel": "传播",
        "center": "中心", "edge": "边缘", "motion": "动作", "follow": "跟随",
        "through": "通过", "crash": "撞击", "crate": "箱子", "montage": "蒙太奇",
        "dissolve": "淡入", "grave": "坟墓", "rest": "休息", "arc": "弧",
        "light": "光", "across": "划过", "line": "线条", "forward": "向前",
        "move": "移动", "macro": "宏", "slide": "滑动", "top": "顶部",
        "down": "向下", "drift": "漂移", "then": "然后", "aerial": "空中",
        "chase": "追逐", "subtle": "微妙", "sway": "摇摆", "dodge": "闪避",
        "low": "低", "angle": "角度",
    }
    
    return word_map.get(word_lower, word)

def split_camel_case(text: str) -> list:
    """将驼峰命名或连写单词分割成单词列表"""
    # 先尝试按大写字母分割（驼峰命名）
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', text)
    if len(words) > 1:
        return [w.lower() for w in words]
    
    # 如果只有一个单词，尝试按常见单词边界分割
    # 使用常见英文单词作为分割点
    text_lower = text.lower()
    common_words = [
        "commission", "weapon", "hire", "cart", "tip", "servant", "throw", "guard",
        "gain", "admission", "receive", "assignment", "plan", "secret", "plot", "strategy",
        "seal", "pact", "show", "strength", "study", "manual", "calm", "troops",
        "city", "lockdown", "report", "discuss", "silver", "wolf", "evaluate", "threat",
        "orders", "march", "out", "assembly", "distribute", "weapons",
        "announce", "victory", "share", "background", "remark", "mystery", "patrol", "air",
        "beast", "tide", "ponder", "sight", "defense", "ready", "present", "token",
        "magical", "scan", "inspect", "buy", "material", "pay", "spirit", "stone",
        "read", "inn", "corner", "visualize", "map", "array", "consider", "heavenly",
        "tribulation", "decide", "training", "destroy", "books",
    ]
    
    # 尝试找到匹配的单词
    words = []
    i = 0
    while i < len(text_lower):
        matched = False
        for word in sorted(common_words, key=len, reverse=True):
            if text_lower[i:].startswith(word):
                words.append(word)
                i += len(word)
                matched = True
                break
        if not matched:
            # 如果无法匹配，尝试匹配单个字符或短单词
            if i < len(text_lower):
                words.append(text_lower[i])
                i += 1
            else:
                break
    
    if len(words) > 1:
        return words
    return [text.lower()]

def translate_action(action: str) -> str:
    """翻译 action 字段"""
    if not action:
        return action
    
    # 检查是否已包含中文
    if re.search(r'[\u4e00-\u9fff]', action):
        return action
    
    # 先检查扩展映射表
    if action in extended_action_map:
        return extended_action_map[action]
    
    # 尝试下划线分割翻译
    if "_" in action:
        parts = action.split("_")
        translated_parts = []
        for part in parts:
            translated = translate_english_word(part)
            if translated:
                translated_parts.append(translated)
        if translated_parts:
            return "".join(translated_parts)
    
    # 尝试分割连写单词（驼峰命名或全小写连写）
    words = split_camel_case(action)
    if len(words) > 1:
        translated_parts = []
        for word in words:
            translated = translate_english_word(word)
            if translated and translated != word:
                translated_parts.append(translated)
        if translated_parts:
            return "".join(translated_parts)
    
    # 如果无法翻译，返回原值
    return action

def translate_motion(motion: str) -> str:
    """翻译 motion 字段"""
    if not motion:
        return motion
    
    # 检查是否已包含中文
    if re.search(r'[\u4e00-\u9fff]', motion):
        return motion
    
    # 先检查扩展映射表
    if motion in extended_motion_map:
        return extended_motion_map[motion]
    
    # 尝试按空格或连字符分割翻译
    words = re.split(r'[\s-]+', motion)
    translated_words = []
    for word in words:
        if not word:
            continue
        # 先尝试直接翻译
        translated = translate_english_word(word)
        if translated and translated != word:
            translated_words.append(translated)
        else:
            # 如果无法直接翻译，尝试分割连写单词
            split_words = split_camel_case(word)
            if len(split_words) > 1:
                for sw in split_words:
                    st = translate_english_word(sw)
                    if st and st != sw:
                        translated_words.append(st)
            else:
                translated_words.append(word)
    
    if translated_words:
        return "".join(translated_words)
    
    # 如果无法翻译，返回原值
    return motion

def process_json_file(json_path: Path):
    """处理单个 JSON 文件"""
    print(f"处理: {json_path.name}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    updated_count = 0
    
    for scene in scenes:
        # 翻译 action
        action = scene.get("action", "")
        if action:
            new_action = translate_action(action)
            if new_action != action:
                scene["action"] = new_action
                updated_count += 1
        
        # 翻译 visual.motion
        visual = scene.get("visual", {}) or {}
        if isinstance(visual, dict):
            motion = visual.get("motion", "")
            if motion:
                new_motion = translate_motion(motion)
                if new_motion != motion:
                    visual["motion"] = new_motion
                    updated_count += 1
                    scene["visual"] = visual
    
    if updated_count > 0:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  更新了 {updated_count} 个字段")
    else:
        print(f"  无需更新")

def main():
    """主函数"""
    base_dir = Path("lingjie/scenes")
    
    # 处理 3-11.json
    for i in range(3, 12):
        json_path = base_dir / f"{i}.json"
        if json_path.exists():
            process_json_file(json_path)
        else:
            print(f"文件不存在: {json_path}")

if __name__ == "__main__":
    main()

