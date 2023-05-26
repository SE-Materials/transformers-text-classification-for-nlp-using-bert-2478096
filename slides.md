# Slides

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/eb6669b5-7d7b-4c37-be24-b28b990b59f3)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/321aa933-2de5-4dab-98da-cefb6b85944f)

## Challenge 1

### Finding number of parameters

How many parameters does the BERT base cased model have (bert-base-cased)? Use the get_model_size function below to help you.

```python
def get_model_size(checkpoint='bert-base-cased'):
  '''
  Usage: 
      checkpoint - this is NLP model with its configuration and its associated weights
      returns the size of the NLP model you want to determine
  '''
  
  model = AutoModel.from_pretrained(checkpoint)
  return sum(torch.numel(param) for param in model.parameters())
```

### Calculating RAM requirements

If you know the number of parameters for a model, how might you be able to determine how much memory is required when running a model inference? (Each parameter is represented as a single precision floating point number)

4 * number of parameters / 10^6 (MB)

If you wanted to run a GPT-3 inference. How much RAM would your infrastructure require.

<hr>

## Text classification in NLP using BERT

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/fbfba926-862f-45a3-a9d0-30fad98d9463)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/a01f2964-7fc1-4578-86e2-9acd84b2a4ba)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/8b9a577d-867a-4195-adee-984a5d9e60b9)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/5ccfbf15-4128-4ea1-a3fe-628ccc3e6d1b)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/6d10af83-b28f-4da4-93a2-3449ef8ec9ad)

## Transfer learning

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/bfe50c22-8b02-4bd9-827f-53a9f4138a77)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/d52d4bc6-90a2-4f77-8505-194a2b98a8a8)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/33dd9900-2cf4-49e3-8046-0fec18a720e8)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/848439b0-f491-4115-ab44-fb00daa63941)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/67de7d8a-6fe0-4fb2-8eb9-2730bfee7054)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/ed6777de-8789-4f41-9086-b6f70ed9acb0)

# Transformer Architecture

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/d54a2e4d-71ec-4cdd-9887-fa8c82f4f944)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/15653295-74b8-4860-8679-f990570a9d58)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/eb4aa3b8-0927-4a36-8a7f-7e8b477368a1)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/a3dd358e-3e98-4179-8077-eb6c61189f7a)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/181e0ccc-614e-4c7e-b587-790d8483f50a)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/0db63314-2f4b-46a9-8b3c-8d54dde11a05)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/ddcc4cf9-4e41-469c-9c44-1c630a5ba9ab)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/fff84a7c-4fd8-47d8-af67-88d84dbf4514)


# Tokenization

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/8ba344bd-4047-4e43-9f5b-211a272f09ca)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/8e1822c8-7a92-4b57-80b7-d749e3d12c2b)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/2816e704-78de-44e0-8247-f66f633fc192)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/1ee46f60-cbb2-4c2d-8a9b-ffe5636f9502)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/6156dd2b-0e4e-4f85-8135-4b9900ed4f41)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/bf520e70-bc6c-4851-a7bf-3691e2804f02)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/f834cd83-e844-4228-8aa3-f7358652c913)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/417d112a-23dc-4747-b48f-57e35112bcbe)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/90743bf4-7785-41cb-9f6d-15b7ab1df8af)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/0419882e-baed-407a-8108-723a4617b892)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/75adf788-25b7-4afd-acfa-246e4ea99a6b)


# Positional encodings and segment embeddings

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/da232b6c-14da-4102-bf67-e8f351cdc95a)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/d8595730-2b79-44eb-bedb-38229ce672db)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/bf7ac685-848e-4f4f-b7f4-87748827568c)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/f38dbdd4-f089-46ed-b977-3a0ccec8184c)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/606aebdc-99ab-4100-8fdc-ed84c5397d5d)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/ef35ab22-b676-4e7e-9be9-328d24426a88)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/7500b0c0-11d2-4180-b01e-c8545ffae697)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/9943a59e-53ce-4676-a319-be57c755d7b8)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/8cf3a418-574b-4fc5-ad75-1b7c1d97f5ab)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/f825a42d-f2b4-454d-94c3-da87d39f1c43)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/889f4f23-8035-4fc2-9924-a5a12d26a2f3)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/abfd7c33-7241-4234-86c1-f0097572d9f7)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/2cf3d13a-df79-4910-9c7c-ae91c010f9d7)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/834b57eb-1cc2-40bf-9944-b89917cf575f)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/ba36c448-0931-4d06-a67c-3fc89d91e6a6)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/41d1f84f-c168-4be5-b4c5-fb8bbdaef9df)


# Self attention

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/4a6393e2-1977-417b-a803-c2c97479d415)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/7478b29c-8d70-45c7-94bf-88ccb2d077dd)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/a4716508-8ef4-4a14-9540-d57a3d0639de)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/c103ff13-5c29-4597-ae27-99917ea73a51)

![image](https://github.com/SE-Materials/transformers-text-classification-for-nlp-using-bert-2478096/assets/8214876/ef80d4e1-88eb-4cca-b79c-cd05e454b184)




