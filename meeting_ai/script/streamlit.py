import streamlit as st
from openai import OpenAI
# 对文章进行总结
from autogen import OpenAIWrapper
import json
from tqdm import tqdm

summary_prompt = """
    # Task
    - Summary all the meeting content, firstly a conclusion of meeting, then some bullet points, for each bullet point, expand the opinions for 500 tokens.
    - The output summary should be in chinese.

    # Output json format
    {
        "summary": "Summary of the meeting content, in chinese"
    }

    # Content
"""

def valid_json_filter(response, **_):
    for text in OpenAIWrapper.extract_text_or_completion_object(response):
        try:
            json.loads(text)
            return True
        except ValueError:
            pass
    return False

def process_audio(filename, openai_api_key):
    client = OpenAI(api_key=openai_api_key)

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=filename
    )

    repeat_seed_list = [1, 2, 3, 4, 5]
    config_list = []
    for seed in repeat_seed_list:
        config = {
            "model": "gpt-3.5-turbo-1106",  # gpt-4-1106-preview gpt-35-turbo-1106
            "api_key": openai_api_key,
            "base_url": "https://api.openai.com/v1",
            'cache_seed': seed,
        }
        config_list.append(config)

    # Azure OpenAI endpoint
    client = OpenAIWrapper(
        config_list=config_list
    )

    msg_context = {
        "summary_prompt": summary_prompt
    }

    # Completion
    response = client.create(
        context=msg_context,
        messages=[{"role": "user", "content": f"{summary_prompt} {transcript.text}"}],
        response_format={"type": "json_object"},
        filter_func=valid_json_filter,
    )

    auto_completion = response.choices[0].message.content

    # 解析JSON结果
    result = json.loads(auto_completion)

    result['transcript'] = transcript.text

    return result

def main():
    st.title("Audio Processing App")

    st.warning("🎈需要先将录音后的文件通过 https://www.aconvert.com/audio 转换为mp3文件，然后上传进行解析")

    # 用户输入 OpenAI 密钥
    openai_api_key = st.text_input("输入你的OpenAI密钥", type="password")

    # 用户选择音频文件
    audio_file = st.file_uploader("上传音频文件", type=["wav", "mp3"])

    if audio_file and openai_api_key:
        # 处理音频文件
        result = process_audio(audio_file, openai_api_key)

        # 显示处理结果
        st.subheader("处理结果:")
        st.write(result)

if __name__ == "__main__":
    main()
