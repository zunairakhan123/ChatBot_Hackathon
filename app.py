import nest_asyncio
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
import os
from groq import Groq
import requests
from bs4 import BeautifulSoup

nest_asyncio.apply()

# --- CONFIGURATION ---
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")  # Set in your HuggingFace Secrets
channel_id = "UCsv3kmQ5k1eIRG2R9mWN"  # @icodeguru0
BASE_URL = "https://icode.guru"

# Initialize Groq client once
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- FUNCTION: Fetch recent video IDs from YouTube channel ---
def get_latest_video_ids(channel_id, max_results=5):
    url = f"https://www.googleapis.com/youtube/v3/search?key={YOUTUBE_API_KEY}&channelId={channel_id}&part=snippet,id&order=date&maxResults={max_results}"
    response = requests.get(url)
    videos = response.json().get('items', [])
    return [v['id']['videoId'] for v in videos if v['id']['kind'] == 'youtube#video']

# --- FUNCTION: Get video transcripts ---
def get_video_transcripts(video_ids):
    all_transcripts = []
    for vid in video_ids:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(vid)
            text = " ".join([t['text'] for t in transcript])
            all_transcripts.append(text)
        except:
            continue
    return all_transcripts

# --- NEW FUNCTION: Scrape icode.guru ---
def scrape_icodeguru(base_url="https://icode.guru", max_pages=5):
    visited = set()
    blocks = []

    def crawl(url):
        if url in visited or len(visited) >= max_pages:
            return
        visited.add(url)
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.content, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            if len(page_text) > 100:
                blocks.append(f"[Source]({url}):\n{page_text[:2000]}")
            for link in soup.find_all("a", href=True):
                href = link['href']
                if href.startswith("/"):
                    href = base_url + href
                if href.startswith(base_url):
                    crawl(href)
        except:
            pass

    crawl(base_url)
    return blocks

# --- FUNCTION: Ask Groq API using official client ---
def ask_groq(context, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Only answer using the given context (YouTube + icode.guru). Provide links if possible."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
    ]
    chat_completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Or the model you have access to
        messages=messages,
    )
    return chat_completion.choices[0].message.content.strip()

# --- STREAMLIT APP ---
def main():
    st.set_page_config(page_title="EduBot - YouTube + iCodeGuru QA", layout="wide")
    st.title("🎓 EduBot for @icodeguru0")
    st.markdown("Ask anything based on the channel’s recent videos and website content from [icode.guru](https://icode.guru).")

    question = st.text_input("💬 Ask your question here:")
    if question:
        with st.spinner("🔍 Fetching videos and transcripts..."):
            video_ids = get_latest_video_ids(channel_id)
            transcripts = get_video_transcripts(video_ids)
            yt_context = "\n\n".join(transcripts)

        with st.spinner("🌐 Scraping icode.guru..."):
            site_blocks = scrape_icodeguru(BASE_URL, max_pages=5)
            site_context = "\n\n".join(site_blocks)

        full_context = yt_context + "\n\n" + site_context

        with st.spinner("🧠 Thinking..."):
            answer = ask_groq(full_context, question)
        st.success(answer)

    st.markdown("---")
    st.caption("Powered by YouTube + iCodeGuru + Groq | Built for @icodeguru0")

if __name__ == "__main__":
    main()
