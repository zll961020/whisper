import torch 
import whisper 
from whisper.tokenizer import get_tokenizer
import os 
import numpy as np
from whisper.decoding import DecodingOptions
import re, string
import json  
from pathlib import Path
import argparse

def strip_punctuation(text: str) -> str:
    # 英文 + 常见中日韩符号，如需更多自行补充
    punctuation = string.punctuation + "，。！？；：【】（）《》‘’“”…—、"
    return re.sub(f"[{re.escape(punctuation)}]", "", text)


def test(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(f'./pretrain/{model_name}.pt').to(device)
    audio_path = os.path.join(os.path.dirname(__file__), "tests/jfk.flac")

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    print(f'result language: {result["language"]}')
    print(f'result text: {result["text"]}')
    print(f'segments: {"".join([s["text"] for s in result["segments"]])}')
    assert result["language"] == "en"
    assert result["text"] == "".join([s["text"] for s in result["segments"]])

    transcription = result["text"].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription

    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
    all_tokens = [t for s in result["segments"] for t in s["tokens"]]
    print(f' all_tokens: {all_tokens}')
    assert tokenizer.decode(all_tokens) == result["text"]
    assert tokenizer.decode_with_timestamps(all_tokens).startswith("<|0.00|>")
    print(f'text: {result["text"]} tokenizer decode: {tokenizer.decode(all_tokens)} ')
    print(f'tokenizer decode with timestamps: {tokenizer.decode_with_timestamps(all_tokens)}')

    timing_checked = False
    for segment in result["segments"]:
        for timing in segment["words"]:
            assert timing["start"] < timing["end"]
            if timing["word"].strip(" ,") == "Americans":
                assert timing["start"] <= 1.8
                assert timing["end"] >= 1.8
                timing_checked = True

    assert timing_checked

def infer(audio_path, model_name: str='large-v3'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(f'./pretrain/{model_name}.pt').to(device)

    # load audio and pad/trim it to fit 30 seconds
    # audio = whisper.load_audio(audio_path)
    # audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    # mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # # detect the spoken language
    # _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # # decode the audio
    # options = whisper.DecodingOptions()
    # result = whisper.decode(model, mel, options)

    # print the recognized text
    # print(result.text)
    result =  model.transcribe(audio_path, language='zh', temperature=0.0, word_timestamps=True, initial_prompt='浩鲸科技,云、大数据与人工智能。', carry_initial_prompt=True)#initial_prompt='浩鲸科技，云、大数据与人工智能。', carry_initial_prompt=True
    print(f'result language: {result["language"]}')
    print(f'result text: {result["text"]}')
    #print(f'segments: {result["segments"]}')


def infer_attention_beam_search(model, audio_path, language: str='zh', temperature: float=0.0, beam_size: int=5):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    result = model.transcribe(audio_path, language=language, temperature=temperature, beam_size=beam_size, without_timestamps=True)
    # print(f'result language: {result["language"]}')
    # print(f'result text: {result["text"]}')
    # print(f'result text without punctuation: {strip_punctuation(result["text"])}')
    #print(f'segments: {result["segments"]}')
    return strip_punctuation(result["text"])

# ---------- 主逻辑 ----------
def main(args):
    """批量转录文本 加断点恢复 
    """
    
    model = whisper.load_model(f'./pretrain/{args.model_name}.pt').to(args.device)
    # ------------ 恢复机制 ------------
    processed_keys = set()
    if Path(args.output).is_file():
        with open(args.output, "r", encoding="utf-8") as f_exist:
            for line in f_exist:
                line = line.strip()
                if not line:
                    continue
                # 输出格式:  key<space>text
                processed_keys.add(line.split(maxsplit=1)[0])
        print(f"[Resume] 已读取 {len(processed_keys)} 条已完成数据，运行时将跳过。")

    # 统计
    total   = len(processed_keys) 
    ok      = len(processed_keys)   # 已完成也算成功
    fail    = 0

    # 根据是否恢复决定写文件模式
    file_mode = "a" if processed_keys else "w"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 打开文件
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, file_mode, encoding="utf-8") as fout:
        for line in fin:
            try:
                item = json.loads(line.strip())
                key = item["key"]
                wav_path = item["wav"]
                 # ---- 已处理则跳过 ----
                if key in processed_keys:
                    if args.verbose:
                        print(f"[Skip] {key} 已存在，跳过。")
                    continue
                total += 1
                if not Path(wav_path).is_file():
                    raise FileNotFoundError(wav_path)

                # Whisper 推理
                
                text = infer_attention_beam_search(model, wav_path, 
                                                   args.language, args.temperature, args.beam_size)

                fout.write(f"{key} {text}\n")
                ok += 1
                if args.verbose:
                    print(f"[{ok}/{total}] {key}: {text}")
            except Exception as e:
                fail += 1
                print(f"[ERROR] line {total}: {e}")

    print(f"\nDone. success={ok}, fail={fail}, total={total}")
    print(f"结果已写入: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Batch transcribe wav files listed in a json-lines txt."
    )
    parser.add_argument("input",  help="wav 列表 txt（JSON-Lines）")
    parser.add_argument("output", help="输出 txt")
    parser.add_argument("--model",   default="large-v3", help="Whisper 模型大小")
    parser.add_argument("--device",  default="cuda" if whisper.torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("--language", default="zh", help="音频语言，用于跳过自动检测")
    parser.add_argument("--temperature", default=0.0, help="温度")
    parser.add_argument("--beam_size", default=5, help="beam_size")
    parser.add_argument("--verbose", action="store_true", help="打印逐条结果")
    args = parser.parse_args()

    main(args)
    

