import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import requests

class webRetriever(object):
    def __init__(self, session):
        self.session = session
    
    def read_web(self, url):
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').text
        loader = WebBaseLoader(
            web_paths=(url,),
            session=self.session,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        
        return docs, title
    
    def split_text(self, docs, url, title):
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n\n\n"],chunk_size=2000, chunk_overlap=200, length_function=len,
                                                       is_separator_regex=False)
        splits = text_splitter.split_documents(docs)

        for split in splits:
            split.metadata["url"] = url
            split.metadata["title"] = title

        return splits

def insert_vectorstore(splits, vectorstore):
    # insert with title as metadata
    if vectorstore:
        vectorstore.add_documents(splits)
    else:
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

    return vectorstore
    
def web_retrieve(url_list):
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    web_retriever = webRetriever(session)

    docs = {}
    vectorstore = None
    for paths in url_list:
        doc, title = web_retriever.read_web(paths)
        splits = web_retriever.split_text(doc, paths, title)
        if splits:
            vectorstore = insert_vectorstore(splits, vectorstore)

    return vectorstore

retriever = web_retrieve([
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]).as_retriever(search_type="similarity", search_kwargs={'k': 6})



