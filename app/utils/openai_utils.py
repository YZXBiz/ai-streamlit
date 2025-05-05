from pydantic_settings import BaseSettings
import os
from azure.identity import CertificateCredential
from openai import AzureOpenAI


class Settings(BaseSettings):
    ROOT_DIR: str = "/Users/c839755/Desktop/Projects/Tutorial-Codebase-Knowledge"

    DEPLOYMENT_NAME: str = "gpt-4o"
    OPENAI_API_TYPE: str = "azure_ad"
    OPENAI_API_VERSION: str = "2024-05-01-preview"

    EMBEDDING_MODEL: str = "exai-texemb"

    AZURE_TENANT_ID: str = "fabb61b8-3afe-4e75-b934-a47f782b8cd7"
    AZURE_CLIENT_ID: str = "ca7f2556-0a9c-48f9-bc1c-cffb6b434a5a"
    SP_CERTIFICATE_SECRET: str = "Frontstore#2025"

    @property
    def OPENAI_CRED_PATH(self) -> str:
        return os.path.join(self.ROOT_DIR, "credentials/openai_credentials.pfx")

    @property
    def OPENAI_API_KEY(self) -> CertificateCredential:
        credential = CertificateCredential(
            tenant_id=self.AZURE_TENANT_ID,
            client_id=self.AZURE_CLIENT_ID,
            certificate_path=self.OPENAI_CRED_PATH,
            password=self.SP_CERTIFICATE_SECRET,
        )
        openai_api_key = credential.get_token("https://cognitiveservices.azure.com/.default").token

        return openai_api_key


SETTINGS = Settings()

llm = AzureOpenAI(
    azure_endpoint="https://corpgenaiipocuse2exai.openai.azure.com/",
    azure_ad_token=SETTINGS.OPENAI_API_KEY,
    api_version=SETTINGS.OPENAI_API_VERSION,
)
