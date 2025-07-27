
import nest_asyncio
import streamlit as st
import os
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

nest_asyncio.apply()

# --- CONFIGURATION ---
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
channel_id="UCsv3kmQ5k1eIRG2R9mWN-QA"  #channelId

BASE_URL = "https://icode.guru"

groq_client = Groq(api_key=GROQ_API_KEY)
embedding_function = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("icodeguru_knowledge", embedding_function=embedding_function)

# --- Search persistent vector DB ---
def search_vector_data(query):
    results = collection.query(query_texts=[query], n_results=3)
    if results and results["documents"]:
        return "\n\n".join([doc for doc in results["documents"][0]])
    return None

# --- Fetch recent video IDs from YouTube channel ---
def get_latest_video_ids(channel_id, max_results=5):
    url = f"https://www.googleapis.com/youtube/v3/search?key={YOUTUBE_API_KEY}&channelId={channel_id}&part=snippet,id&order=date&maxResults={max_results}"
    response = requests.get(url)
    videos = response.json().get('items', [])
    
    valid_videos = []
    for v in videos:
        if v['id']['kind'] == 'youtube#video':
            title = v['snippet']['title']
            channel_title = v['snippet']['channelTitle']
            video_id = v['id']['videoId']
            if "icodeguru" in channel_title.lower():
                valid_videos.append((video_id, title))
    return valid_videos

# --- Get video transcripts ---
def get_video_transcripts(video_info):
    results = []
    for vid, title in video_info:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(vid)
            text = " ".join([t['text'] for t in transcript])
            video_link = f"https://www.youtube.com/watch?v={vid}"
            results.append({
                "video_id": vid,
                "title": title,
                "link": video_link,
                "transcript": text
            })
        except:
            continue
    return results

# --- Scrape icode.guru ---
def scrape_icodeguru(base_url=BASE_URL, max_pages=5):
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
                blocks.append(f"[{url}]({url}):\n{page_text[:1500]}")
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

# --- Ask Groq ---
def ask_groq(context, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always provide relevant video and website links if possible."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer (include links):"}
    ]
    chat_completion = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
    )
    return chat_completion.choices[0].message.content.strip()

#--- STREAMLIT APP ---
def main():
    st.set_page_config(page_title="EduBot for iCodeGuru", layout="wide")
    st.title("🎓 EduBot for @icodeguru0")
    st.markdown("Ask anything based on the latest YouTube videos and website content of [icode.guru](https://icode.guru).")

    user_question = st.text_input("💬 Ask your question:")

    if user_question:
        # Try vector DB first
        vector_context = search_vector_data(user_question)
        if vector_context:
            with st.spinner("🧠 Answering from knowledge base..."):
                answer = ask_groq(vector_context, user_question)
                st.success(answer)
        else:
            # Fallback to real-time data
            with st.spinner("📺 Fetching YouTube videos..."):
                video_info = get_latest_video_ids(channel_id, max_results=5)
                transcripts = get_video_transcripts(video_info)

            yt_context = ""
            relevant_links = []
            for vid in transcripts:
                yt_context += f"\n\n[Video: {vid['title']}]({vid['link']}):\n{vid['transcript'][:1500]}"
                if user_question.lower() in vid['transcript'].lower():
                    relevant_links.append(vid['link'])

            with st.spinner("🌐 Scraping icode.guru..."):
                site_blocks = scrape_icodeguru(BASE_URL, max_pages=5)
                site_context = "\n\n".join(site_blocks)

            full_context = yt_context + "\n\n" + site_context

            with st.spinner("🧠 Thinking..."):
                answer = ask_groq(full_context, user_question)
                st.success(answer)

            if relevant_links:
                st.markdown("### 🔗 Related YouTube Links")
                for link in relevant_links:
                    st.markdown(f"- [Watch Video]({link})")

    st.markdown("---")
    st.caption("Powered by YouTube, iCodeGuru, and Groq")

if __name__ == "__main__":
    main()
