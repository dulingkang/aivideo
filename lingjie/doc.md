下面把你问的 **A 与 B** 两部分内容，**非常清晰地总结**出来，便于你直接写入 JSON 自动化流水线使用。

---

# **A. face_style_auto 全字段可选值（最终建议版）**

这些值是为“**国漫 / 动漫视频生成自动化流水线**”专门设计的，既通用又便于后续 Stable Diffusion / AnimateDiff 自动解析，不会出现不可控漂移。

---

## **✔ 1）face_style_auto.expression（表情）完整可选值**

（保持 10 个以内最常用、最易控制的值）

| 表情值            | 描述         |
| -------------- | ---------- |
| **neutral**    | 中性表情（默认最稳） |
| **focused**    | 专注、凝神      |
| **serious**    | 严肃、冷静      |
| **calm**       | 稳定、淡定      |
| **alert**      | 提防、观察四周    |
| **determined** | 坚定、意志强     |
| **happy_soft** | 微微开心、嘴角上扬  |
| **smirk**      | 一侧嘴角微挑（自信） |
| **angry_low**  | 生气但不夸张     |
| **tired_soft** | 轻微疲惫       |

> **建议：在打斗剧情里只用 neutral, focused, serious, determined 保持一致性。**

---

## **✔ 2）face_style_auto.lighting（光照）完整可选值**

| 光照值                | 描述             |
| ------------------ | -------------- |
| **bright_normal**  | 明亮常规光（户外・默认）   |
| **soft**           | 柔光、弱对比（对话、特写）  |
| **dramatic**       | 强对比、戏剧光（大战、紧张） |
| **rim_light**      | 轮廓光（修仙风常用）     |
| **dark_lowkey**    | 低光（夜晚场景）       |
| **warm_indoor**    | 室内暖色光          |
| **cool_moonlight** | 月光冷色           |

> **建议：保持 4~5 个常用光照，效果更稳定。**

---

## **✔ 3）face_style_auto.detail（细节程度）可选值**

| 值                     | 描述              |
| --------------------- | --------------- |
| **natural**           | 自然细节（默认）        |
| **soft_concentrated** | 柔化但保持关键五官（国漫画风） |
| **detailed**          | 增加细节清晰度（用于超分前）  |
| **subtle**            | 极简细节（背景模糊时使用）   |
| **cinematic**         | 影视级细节（对大场面稳定）   |

---

# **B. visual.motion / action / visual.character_pose 的关系（最重要的部分）**

我帮你总结成一张表，你完全照这个做即可：

---

## **✔ visual.character_pose（人物静态姿势）**

* **描述模型生成该帧的角色姿态**
* 对应“单帧姿势控制”
* 适用于：InstantID、ControlNet-OpenPose、单帧图生成

示例：

```json
"visual": {
  "character_pose": "standing, right hand holding a sword, slight lean forward"
}
```

---

## **✔ action（人物正在做的动作 / 情节）**

* **叙事层动作，不完全等于单帧姿势**
* 用于给旁白 + 给多帧动画（AnimateDiff）做“动作意图引导”

示例：

```json
"action": "Han Li steps forward and slashes with green light"
```

---

## **区别总结（一句话）**

* **character_pose 是“身体摆成什么形状”**
* **action 是“角色正在做什么”**

> 在自动化动画流水线中：
> **character_pose → 控制单帧生成稳定**
> **action → 控制动画帧序列的趋势**

---

## **✔ visual.motion（镜头运动）格式规范**

为了便于你流水线解析，我建议使用“**结构化 + 简称**”混合格式：

### **推荐格式（结构化、易解析）**

```json
"visual": {
  "motion": {
    "type": "pan", 
    "direction": "left_to_right",
    "speed": "slow"
  }
}
```

### motion.type 可选值

| 值            | 说明       |
| ------------ | -------- |
| **static**   | 静止镜头     |
| **pan**      | 水平移动     |
| **tilt**     | 垂直移动     |
| **push_in**  | 推镜头（向前）  |
| **pull_out** | 拉镜头（向后）  |
| **orbit**    | 环绕角色     |
| **shake**    | 轻微抖动（战斗） |
| **follow**   | 跟拍       |

### motion.speed 可选值

* slow
* medium
* fast

### motion.direction 可选值

* left_to_right
* right_to_left
* up
* down
* forward
* backward
* around

---

## **visual.motion 是否必须？**

* **不必须**
* 推荐只在重要镜头（开场、战斗、大景）加入
* 其余用 `"type": "static"` 保持高稳定度

---

# **C. 最后总结（你直接照这个使用即可）**

### **你需要填写的字段控制逻辑：**

| 字段                    | 控制对象     | 用途          |
| --------------------- | -------- | ----------- |
| face_style_auto       | 角色“脸”的风格 | 保证主角一致性     |
| visual.character_pose | 单帧姿势     | 控制图像稳定性     |
| action                | 动作文本     | 控制动画趋势 & 旁白 |
| visual.motion         | 镜头运动     | 控制最终视频的镜头语言 |
