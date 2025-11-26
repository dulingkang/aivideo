# JSON场景优化说明

## 优化原则

### 1. 纯背景场景（无人物）
- **composition**: 明确添加 `no person, no character`，强调关键物体（如卷轴、太阳、月亮）
- **prompt**: 明确排除人物，强调关键物体
- **character_pose**: 保持为空字符串 `""`

### 2. 有角色场景
- **character_pose**: 明确指定 `facing camera, front view` 确保人物面向镜头
- **prompt**: 在描述中添加 `facing camera, front view` 确保正面朝向
- **composition**: 包含角色动作和朝向信息

### 3. Prompt优化
- **精准描述**: 去除冗余，保留核心要素
- **明确朝向**: 有角色时明确 `facing camera, front view`
- **排除人物**: 纯背景场景明确 `no person, no character`
- **强调物体**: 关键物体（卷轴、太阳、月亮、鸟等）明确标注

## 优化示例对比

### 场景0（纯背景 - 卷轴）

**优化前**:
```json
"composition": "Immortal realm sky veiled in clouds,golden scroll unfurling"
"prompt": "Xianxia opening,mist-shrouded immortal sky,golden scroll unfurling,spirit particles drifting,wide cinematic shot,film-grade lighting,4K"
```

**优化后**:
```json
"composition": "Golden scroll unfurling in immortal realm sky,veiled in clouds,spirit light shimmering,no person,no character,scroll is the main element"
"prompt": "Xianxia opening,golden scroll unfurling prominently,mist-shrouded immortal sky,spirit particles drifting,wide cinematic shot,no person,no character,film-grade lighting,4K"
```

**改进点**:
- ✅ 卷轴放在composition最前面，强调为主要元素
- ✅ 明确添加 `no person, no character`
- ✅ prompt中强调 `golden scroll unfurling prominently`

### 场景1（有角色 - 韩立躺着）

**优化前**:
```json
"character_pose": "Motionless,feeling the heat"
"prompt": "Han Li lying on gray-green sand,motionless,feeling the ground's dry heat."
```

**优化后**:
```json
"character_pose": "Lying flat on back,motionless,facing upward,facing camera"
"prompt": "Han Li lying motionless on gray-green sand,top-down view,face visible,facing upward,feeling the ground's dry heat,calm expression"
```

**改进点**:
- ✅ 明确指定 `facing camera` 和 `facing upward`
- ✅ 添加 `face visible` 确保人脸可见
- ✅ 添加 `top-down view` 明确镜头角度

### 场景13（有角色 - 韩立战斗）

**优化前**:
```json
"character_pose": ""
"prompt": "The sand spins at his lips while the monstrous birds close to ten-plus zhang."
```

**优化后**:
```json
"character_pose": "Facing camera,front view,sand spinning at lips,focused and determined expression"
"prompt": "Han Li facing camera,front view,the sand spins at his lips while the monstrous birds close to ten-plus zhang,medium shot,focused expression"
```

**改进点**:
- ✅ 明确指定 `facing camera, front view`
- ✅ 添加表情描述 `focused expression`
- ✅ 明确镜头类型 `medium shot`

## 关键优化字段

### composition字段
- **纯背景场景**: 添加 `no person, no character`，强调关键物体
- **有角色场景**: 包含角色动作和朝向信息

### character_pose字段
- **纯背景场景**: 保持为空 `""`
- **有角色场景**: 必须包含 `facing camera, front view` 或明确的朝向描述

### prompt字段
- **纯背景场景**: 添加 `no person, no character`，强调关键物体
- **有角色场景**: 添加 `facing camera, front view`，明确表情和镜头类型

## 优化检查清单

- [ ] 纯背景场景是否明确排除人物？
- [ ] 有角色场景是否明确指定朝向（facing camera）？
- [ ] 关键物体（卷轴、太阳、月亮等）是否在prompt中强调？
- [ ] prompt是否精准，去除冗余？
- [ ] character_pose是否在有角色时明确指定？
- [ ] composition是否清晰描述场景核心要素？

