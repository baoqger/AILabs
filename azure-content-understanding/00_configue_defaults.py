import os

from dotenv import load_dotenv
from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

load_dotenv()


def main() -> None:
    # Create a ContentUnderstandingClient
    # You can authenticate using either DefaultAzureCredential (recommended) or an API key.
    # DefaultAzureCredential will look for credentials in the following order:
    # 1. Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
    # 2. Managed identity (for Azure-hosted applications)
    # 3. Azure CLI (az login)
    # 4. Azure Developer CLI (azd login)
    endpoint = os.environ["AZURE_AI_ENDPOINT"]
    key = os.getenv("AZURE_AI_API_KEY")
    credential = AzureKeyCredential(key)

    client = ContentUnderstandingClient(endpoint=endpoint, credential=credential)

    # [START update_defaults]
    # Get deployment names from environment variables
    gpt_4_1_deployment = os.getenv("GPT_4_1_DEPLOYMENT")
    gpt_4_1_mini_deployment = os.getenv("GPT_4_1_MINI_DEPLOYMENT")
    text_embedding_3_large_deployment = os.getenv("TEXT_EMBEDDING_3_LARGE_DEPLOYMENT")

    # Check if required deployments are configured
    missing_deployments = []
    if not gpt_4_1_deployment:
        missing_deployments.append("GPT_4_1_DEPLOYMENT")
    if not gpt_4_1_mini_deployment:
        missing_deployments.append("GPT_4_1_MINI_DEPLOYMENT")
    if not text_embedding_3_large_deployment:
        missing_deployments.append("TEXT_EMBEDDING_3_LARGE_DEPLOYMENT")

    if missing_deployments:
        print("⚠️  Missing required environment variables:")
        for deployment in missing_deployments:
            print(f"   - {deployment}")
        print("\nPlease set these environment variables and try again.")
        print(
            "The deployment names should match the models you deployed in Microsoft Foundry."
        )
        return

    # Map your deployed models to the models required by prebuilt analyzers
    # The dictionary keys are the model names required by the analyzers, and the values are
    # your actual deployment names. You can use the same name for both if you prefer.
    # At this point, all deployments are guaranteed to be non-None due to the check above
    assert gpt_4_1_deployment is not None
    assert gpt_4_1_mini_deployment is not None
    assert text_embedding_3_large_deployment is not None
    model_deployments: dict[str, str] = {
        "gpt-4.1": gpt_4_1_deployment,
        "gpt-4.1-mini": gpt_4_1_mini_deployment,
        "text-embedding-3-large": text_embedding_3_large_deployment,
    }

    print("Configuring model deployments...")
    updated_defaults = client.update_defaults(model_deployments=model_deployments)

    print("Model deployments configured successfully!")
    if updated_defaults.model_deployments:
        for model_name, deployment_name in updated_defaults.model_deployments.items():
            print(f"  {model_name}: {deployment_name}")
    # [END update_defaults]

    # [START get_defaults]
    print("\nRetrieving current model deployment settings...")
    defaults = client.get_defaults()

    print("\nCurrent model deployment mappings:")
    if defaults.model_deployments and len(defaults.model_deployments) > 0:
        for model_name, deployment_name in defaults.model_deployments.items():
            print(f"  {model_name}: {deployment_name}")
    else:
        print("  No model deployments configured yet.")
    # [END get_defaults]


if __name__ == "__main__":
    main()