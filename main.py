from config import client, deployment_name
response= client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in two simple sentences."

        }
    ],
    max_tokens=100,
)
print(response.choices[0].message.content)