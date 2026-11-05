# 多任务输入兼容

评估仍为六项 VBench 技术分、三项裁判分和原有最终分。素材只作为三项裁判的上下文；不增加条件遵循分，不读取 metadata.json。

## 批量目录

cases_root 可以指向多个维度的父目录，也可以直接指向一个维度目录。

| 模式 | 必需素材 |
| --- | --- |
| t2va | 无参考素材 |
| f2va | input_01 图片，首帧输入 |
| l2va | input_01 图片，尾帧输入 |
| fl2va | input_01、input_02 图片，分别为首尾帧输入 |
| r2va | 至少一个 input_XX 图片、ref_video_XX 视频或 ref_audio_XX 音频 |

支持 png/jpg/jpeg/webp、mp4/mov/mkv/webm/avi、wav/mp3/flac/m4a/aac/ogg。相同类别和编号不能重复。配置的 video_filenames 只指定待评模型成片，不要填参考视频。模式目录是唯一的模式依据，不根据提示词重新分类。

提示词按 prompt_filename 读取，素材编号按文件名读取。明确引用了缺失编号会报告输入问题，不自动猜测编号。f2va/l2va/fl2va 只接受规定的端点图片；更多参考素材的任务应使用 r2va。任何 skip 或缺失案例使批量命令返回非零退出码，同时继续处理其他有效案例。

复制 eval_multimodal.example.yaml 后配置自己的输入根目录、成片文件名和 case_ids。示例使用现有 D14 数据，不复制素材。当前 D14 中 r2va/140603 的原文引用图片5但只有两张图片，必须先消除编号歧义；不会借用增强稿或 metadata 猜测。

## 裁判上下文

参考图片按长边缩放并转 JPEG；参考视频在全长均匀抽帧（默认最多8帧），标注来源编号和时间戳；首尾帧任务另附成片实际第一帧/最后一帧。模型只对生成成片打原有三分。允许提示词指定的编辑和场景变化，参考素材自身不是待评分画面。

图片总数超过 max_context_images（默认64）报错，不静默丢弃素材。可以调整 reference_max_frames 或换用支持更大上下文的服务。抽帧只能提供采样证据，不能保证捕捉所有短暂事件。

音频含独立参考音频和参考视频中的音轨。解码成16kHz单声道 WAV，同时向裁判提供成片音轨；不存在成片音轨时明确告知裁判。默认 VideoScore2 服务未经音频能力验证，因此 audio_input_format=disabled：此类案例可以获得技术分，裁判记录错误，不生成完整九项结果或最终分。

使用经过验证的音频裁判服务时，在 judge.base_url/model 配置它，并按服务协议将 audio_input_format 设为 input_audio 或 audio_url。audio_max_seconds 默认60，超长音频明确失败，不静默截断。该传输适配不代表已验证任何服务的音频理解能力。三分量表不增加音色、音乐或声音质量分。

## 单条与 JSONL

单条：`python -m pipeline --mode fl2va --video generated.mp4 --prompt '...' --reference-image input_01.png --reference-image input_02.png --output outputs/demo`。

参考生可重复传 --reference-image、--reference-video、--reference-audio，分别按传入顺序编号。

JSONL 每行使用唯一 id、mode、prompt、video、references。例如：

```json
{"id":"r2va-001","mode":"r2va","prompt":"图1中的人物执行参考视频1的动作","video":"generated.mp4","references":[{"type":"image","index":1,"path":"input_01.png"},{"type":"video","index":1,"path":"ref_video_01.mp4"}]}
```

JSONL 的素材和视频相对路径以清单所在目录为基准。命令行路径以工作目录为基准。旧 JSONL 不指定 mode/references 时仍按 t2va 处理。

## 结果与缓存

技术计算与最终分公式不变。纯文生批次维持原时长分组；其他批次按模式和时长分组。非文生输出 id 为维度__模式__case，防止同编号覆盖；单任务旧文生路径保留。多维度有重复文生编号时同样使用命名空间。单条/JSONL 的 id 必须由调用者保证唯一。

批量缓存比对提示词、模式、视频和素材内容哈希及配置和协议版本；旧缓存重新计算。结果记录素材路径和裁判实际收到的证据信息。配置或素材变化不会复用旧完整分。

技术指标可能对有意场景变化或逐格动画产生偏低分，这次不改变计算或权重。不同裁判上下文、服务或协议版本的分数需要经过回归校准后比较。
