import streamlit as st
from openai import OpenAI
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageColor
import json
import io
from mistralai import Mistral
import requests
from PIL import Image
import os
import time
import anthropic
import cohere

# Additional Colors
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def plot_bounding_boxes(image, bounding_boxes_json):
    """Draws bounding boxes on an image using PIL."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Color List
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown',
        'gray', 'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy',
        'maroon', 'teal', 'olive', 'coral', 'lavender', 'violet', 'gold', 'silver'
    ] + additional_colors

    # Parse bounding box data
    bounding_boxes = json.loads(bounding_boxes_json)

    for i, box in enumerate(bounding_boxes):
        color = colors[i % len(colors)]
        abs_y1 = int(box["box_2d"][0] / 1000 * height)
        abs_x1 = int(box["box_2d"][1] / 1000 * width)
        abs_y2 = int(box["box_2d"][2] / 1000 * height)
        abs_x2 = int(box["box_2d"][3] / 1000 * width)

        # Swap values if needed
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw bounding box
        draw.rectangle([(abs_x1, abs_y1), (abs_x2, abs_y2)], outline=color, width=4)
        if "label" in box:
            draw.text((abs_x1 + 5, abs_y1 - 10), box["label"], fill=color)

    return img
    
def send_generation_request(
    host,
    params,
):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {'sk-zh1iIlnfsRWRNOwAYQnIvlG4Fah8DR0qU7wAD5VtzS0mmKsg'}"
    }

    # Encode parameters
    files = {}
    image = params.pop("image", None)
    if image is not None:
        files["image"] = (image.name, image.getvalue(), image.type)

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response

def send_async_generation_request(
    host,
    params,
    files = None
):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {'sk-zh1iIlnfsRWRNOwAYQnIvlG4Fah8DR0qU7wAD5VtzS0mmKsg'}"
    }

    if files is None:
        files = {}

    # Encode parameters
    image = params.pop("image", None)
    if image is not None:
        files["image"] = (image.name, image.getvalue(), image.type)
    mask = params.pop("mask", None)


    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    # Process async response
    response_dict = json.loads(response.text)
    generation_id = response_dict.get("id", None)
    assert generation_id is not None, "Expected id in response"

    # Loop until result or timeout
    timeout = int(os.getenv("WORKER_TIMEOUT", 500))
    start = time.time()
    status_code = 202
    while status_code == 202:
        print(f"Polling results at https://api.stability.ai/v2beta/results/{generation_id}")
        response = requests.get(
            f"https://api.stability.ai/v2beta/results/{generation_id}",
            headers={
                **headers,
                "Accept": "*/*"
            },
        )

        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        status_code = response.status_code
        time.sleep(10)
        if time.time() - start > timeout:
            raise Exception(f"Timeout after {timeout} seconds")

    return response
    

def main():
    st.title("Multi-LLM AI Utility")
    
    llm_choice = st.selectbox("Select LLM:", ["OpenAI", "Gemini", "Mistral","Stability AI","Cohere","Claude","Deepseek","LLAMA"])
    
    if llm_choice == "OpenAI":
        use_case = st.selectbox("Select OpenAI Use Case:", [
            "Joke Generator", "Image Generator", "Text Embeddings", "Text-to-Speech", "Audio Transcription"
        ])
        
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        if use_case == "Joke Generator":
            prompt = st.text_input("Enter a prompt for the joke:", "Tell me a programming joke")
            if st.button("Generate Joke"):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(response.choices[0].message.content)
        
        elif use_case == "Image Generator":
            prompt = st.text_input("Enter image description:", "A cute baby sea otter")
            if st.button("Generate Image"):
                response = client.images.generate(prompt=prompt, n=1, size="1024x1024")
                st.image(response.data[0].url)
                
        elif use_case == "Text Embeddings":
            text = st.text_area("Enter text for embedding:")
            if st.button("Generate Embedding"):
                response = client.embeddings.create(model="text-embedding-3-large", input=text)
                st.write(response.data[0].embedding)
                
        elif use_case == "Text-to-Speech":
            text = st.text_area("Enter text to convert to speech:")
            if st.button("Generate Speech"):
                response = client.audio.speech.create(model="tts-1", voice="nova", input=text)
                response.stream_to_file("output.mp3")
                st.audio("output.mp3")
                
        elif use_case == "Audio Transcription":
            audio_file = st.file_uploader("Upload an audio file:")
            if audio_file and st.button("Transcribe Audio"):
                transcription = client.audio.translations.create(model="whisper-1", file=audio_file)
                st.write(transcription.text)
    
    elif llm_choice == "Gemini":
        use_case = st.selectbox("Select Gemini Use Case:", [
            "Text Generator", "Describe Image", "Compare Images", "Object Detection", "Describe Audio"
        ])
        
        client = genai.Client(api_key="AIzaSyDSmkGuwvm0ycSO5XDhmYcN-fQxkrxSU44")
        
        if use_case == "Text Generator":
            prompt = st.text_area("Enter prompt for text generation:")
            if st.button("Generate Text"):
                response = client.models.generate_content(
                    model="gemini-2.0-flash", contents=prompt,
                    config=types.GenerateContentConfig(max_output_tokens=50, temperature=0.1)
                )
                st.write(response.text)
                
        elif use_case == "Describe Image":
            image_file = st.file_uploader("Upload an image:")
            if image_file and st.button("Describe Image"):
                image = Image.open(image_file)
                response = client.models.generate_content(
                    model="gemini-2.0-flash", contents=[image, "Tell me about this instrument"]
                )
                st.write(response.text)
                
        elif use_case == "Compare Images":
            image_file1 = st.file_uploader("Upload first image:")
            image_file2 = st.file_uploader("Upload second image:")
            if image_file1 and image_file2 and st.button("Compare Images"):
                pil_image = Image.open(image_file1)
                b64_image = types.Part.from_bytes(
                    data=image_file2.read(), mime_type="image/jpeg"
                )
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp", contents=["What do these images have in common?", pil_image, b64_image]
                )
                st.write(response.text)
                
        elif use_case == "Object Detection":
            image_file = st.file_uploader("Upload an image:")
            if image_file and st.button("Detect Objects"):
                image = Image.open(image_file)
                response = client.models.generate_content(
                    model="gemini-2.0-flash", contents=["Detect the 2D bounding boxes of the image with labels.", image]
                )
                st.write("Raw Detection Data:", response.text)

                try:
                    detected_img = plot_bounding_boxes(image, response.text)
                    st.image(detected_img, caption="Detected Objects", use_column_width=True)
                except Exception as e:
                    st.error(f"Error processing bounding boxes: {e}")
                
        elif use_case == "Describe Audio":
            audio_file = st.file_uploader("Upload an audio file:", type=["mp3", "wav"])
           
            if audio_file and st.button("Describe Audio"):
                myfile = client.files.upload(file=audio_file)
                response = client.models.generate_content(
                    model="gemini-2.0-flash", contents=["Transcribe the audio clip", myfile]
                )
                st.write(response.text)
				
    elif llm_choice == "Mistral":
        use_case = st.selectbox("Select Mistral Use Case:", [
            "Text Chat", "Text Embeddings", "Image Analysis", "Code Completion"
        ])
        
        client = Mistral(api_key="qatAdWnJ84mx2eYRfyfANRCCSFyeGt0e")
        
        if use_case == "Text Chat":
            prompt = st.text_input("Enter a prompt:")
            if st.button("Generate Response"):
                response = client.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": prompt}])
                st.write(response.choices[0].message.content)
        
        elif use_case == "Text Embeddings":
            text = st.text_area("Enter text for embedding:")
            if st.button("Generate Embedding"):
                response = client.embeddings.create(model="mistral-embed", inputs=[text])
                st.write(response)
        
        elif use_case == "Image Analysis":
            image_file = st.file_uploader("Upload an image:")
            if image_file and st.button("Analyze Image"):
                base64_image = encode_image(image_file)
                response = client.chat.complete(model="pixtral-12b-2409", messages=[{"role": "user", "content": [{"type": "text", "text": "What's in this image?"}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}]}])
                st.write(response.choices[0].message.content)

        elif use_case == "Code Completion":
            prompt = st.text_area("Enter code snippet:")
            suffix = st.text_area("Enter suffix for completion:")
            if st.button("Generate Code"):
                response = client.fim.complete(
                    model="codestral-latest", prompt=prompt, suffix=suffix, temperature=0, top_p=1
                )
                st.code(f"{prompt}\n{response.choices[0].message.content}\n{suffix}", language='python')
                
                
    elif llm_choice == "Stability AI":
        mode = st.selectbox("Select Stabilty Use Case:", [
            "Creative Upscaler", "Inpainting","Search-and-Recolor", "Search-and-Replace", "Text-to-Image Generation"
        ])

        # Common parameters
        prompt = st.text_input("Enter Prompt:")
        negative_prompt = st.text_input("Negative Prompt (Optional):", "")
        seed = st.number_input("Seed:", min_value=0, value=0)
        output_format = st.selectbox("Output Format:", ["jpeg", "png", "webp"])

        if mode == "Creative Upscaler":
            image = st.file_uploader("Upload an image:")
            creativity = st.slider("Creativity:", 0.0, 1.0, 0.3)
            if image:
                if st.button("Upscale Image"):
                    params = {"image" : image,"prompt": prompt, "negative_prompt": negative_prompt, "seed": seed, "creativity": creativity, "output_format": output_format}
                    host = f"https://api.stability.ai/v2beta/stable-image/upscale/creative"
                    response = send_async_generation_request(
                        host,
                        params
                    )
                    generated = f"generated_{seed}.{output_format}"
                    with open(generated, "wb") as f:
                        f.write(response.content)              
                    st.image(generated, caption="Generated Image", use_container_width=True)

        elif mode == "Inpainting":
            image = st.file_uploader("Upload Mask Image", type=["png", "jpeg", "jpg"])
            if image:
                if st.button("Inpaint Image"):
                    host = f"https://api.stability.ai/v2beta/stable-image/edit/inpaint"

                    params = {"image" : image, "prompt": prompt, "negative_prompt": negative_prompt, "seed": seed, "mode": "mask", "output_format": output_format}
                   
                    response = send_generation_request(
                        host,
                        params
                    )
                    generated = f"generated_{seed}.{output_format}"
                    with open(generated, "wb") as f:
                        f.write(response.content)              
                    st.image(generated, caption="Generated Image", use_container_width=True)

        elif mode == "Search-and-Recolor":
            image = st.file_uploader("Upload Mask Image", type=["png", "jpeg", "jpg"])

            if image:
             
                select_prompt = st.text_input("Select Object to Modify:")
                grow_mask = st.slider("Grow Mask Size:", 0, 10, 3)
                if st.button("Recolor Image"):
                    host = f"https://api.stability.ai/v2beta/stable-image/edit/search-and-recolor"
                    params = {"image" : image,"grow_mask" : grow_mask,"prompt": prompt, "negative_prompt": negative_prompt, "seed": seed, "grow_mask": grow_mask, "mode": "search", "output_format": output_format, "select_prompt": select_prompt}
                    response = send_generation_request(
                        host,
                        params
                    )
                    generated = f"generated_{seed}.{output_format}"
                    with open(generated, "wb") as f:
                        f.write(response.content)              
                    st.image(generated, caption="Generated Image", use_container_width=True)

        elif mode == "Search-and-Replace":
            image = st.file_uploader("Upload Mask Image", type=["png", "jpeg", "jpg"])

            if image:
                search_prompt = st.text_input("Object to Replace:")
                if st.button("Replace Object"):
                    host = f"https://api.stability.ai/v2beta/stable-image/edit/search-and-replace"
                    params = {"image" : image,"prompt": prompt, "negative_prompt": negative_prompt, "seed": seed, "mode": "search", "output_format": output_format, "search_prompt": search_prompt}
                    response = send_generation_request(
                        host,
                        params
                    )
                    generated = f"generated_{seed}.{output_format}"
                    with open(generated, "wb") as f:
                        f.write(response.content)              
                    st.image(generated, caption="Generated Image", use_container_width=True)

        elif mode == "Text-to-Image Generation":
            aspect_ratio = st.selectbox("Aspect Ratio:", ["21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"])
            if st.button("Generate Image"):
                host = f"https://api.stability.ai/v2beta/stable-image/generate/sd3"
                params = {"prompt": prompt, "negative_prompt": negative_prompt, "aspect_ratio": aspect_ratio, "seed": seed, "output_format": output_format, "model": "sd3.5-large", "mode": "text-to-image"}
                response = send_generation_request(
                    host,
                    params
                )
                generated = f"generated.{output_format}"
                with open(generated, "wb") as f:
                    f.write(response.content)              
                st.image(generated, caption="Generated Image", use_container_width=True)


    elif llm_choice == "Claude":
    ## Claude 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

        use_case = st.selectbox("Select Claude Use Case:", [
            "Text Chat"
        ])
        
        client = anthropic.Anthropic(api_key='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')
        
        if use_case == "Text Chat":
            text = st.text_input("Enter a text:")
            if st.button("Generate Response"):
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    temperature=0,
                    system="You are a world-class poet. Respond only with short poems.",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": text
                                }
                            ]
                        }
                    ]
                )
                st.write(response.choices[0].message.content)

    elif llm_choice == "Deepseek":
    ## Claude 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

        use_case = st.selectbox("Select Deepseek Use Case:", [
            "Text Chat"
        ])
        
        client = OpenAI(api_key="sk-6c1437bfd169496497926b3909f42e82", base_url="https://api.deepseek.com")

        if use_case == "Text Chat":
            text = st.text_input("Enter a text:")
            if st.button("Generate Response"):
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": text},
                    ],
                    stream=False
                )

                st.write(response.choices[0].message.content)       

    elif llm_choice == "Cohere":
    ## Claude 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

        use_case = st.selectbox("Select Deepseek Use Case:", [
            "Text Chat"
        ])
        
        co = cohere.ClientV2("w1XUNo0wLCqhZFiqsftBI1ca7fFQnq18PvQX7Mjp") 
        if use_case == "Text Chat":
            text = st.text_input("Enter a text:")
            if st.button("Generate Response"):
                response = co.chat(
                    model="command-r-plus-08-2024",
                    messages=[
                        {
                            "role": "user",
                            "content": text,
                        }
                    ],
                )

                st.write(response.message.content[0].text)                   

if __name__ == "__main__":
    main()
