from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from langchain_community.retrievers import AzureCognitiveSearchRetriever
from langchain_core.documents import Document
from typing import List


class CustomRetriever(AzureCognitiveSearchRetriever):
    metadata_ready = False
    metadata_storage = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = self._search(query)
        self.metadata_storage = [
            Document(page_content=result.pop(self.content_key), metadata=result)
            for result in search_results
        ]
        self.metadata_ready = True
        return self.metadata_storage
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = await self._asearch(query)
        self.metadata_storage = [
            Document(page_content=result.pop(self.content_key), metadata=result)
            for result in search_results
        ]
        return self.metadata_storage

   