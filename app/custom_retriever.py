from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from langchain_core.retrievers import AzureCognitiveSearchRetriever
from langchain_core.documents import Document
from typing import List


class CustomRetriever(AzureCognitiveSearchRetriever):
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = self._search(query)
        print(search_results)
        return [
            Document(page_content=result.pop(self.content_key), metadata=result)
            for result in search_results
        ]
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = await self._asearch(query)
        print(search_results)
        return [
            Document(page_content=result.pop(self.content_key), metadata=result)
            for result in search_results
        ]

