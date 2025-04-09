import json
import aiohttp
import asyncio
import aiofiles
from pathlib import Path
import time
import os
import argparse
import shutil
from tqdm.asyncio import tqdm

TEMP_DIR="temp"
MAX_RETRIES = 20
INITIAL_TIMEOUT = 400

async def main(api_url, api_key, input_file,  concurrency,model_name):
    start_time = time.time()
    try:
        print("1. Initializing...")
        os.makedirs(TEMP_DIR, exist_ok=True)

        print("2. Preparing queue...")
        queue = await prepare_queue(input_file)

        print(f"3. Launching {concurrency} worker coroutines...")
        progress_bar = tqdm(total=queue.qsize(), desc="Processing", unit="task")
        workers = [
            asyncio.create_task(process_item(queue, api_url, api_key, input_file, model_name, progress_bar))
            for _ in range(concurrency)
]
        await queue.join()
        print("4. Queue processing completed.")
        merged_data = []

        # Combination
        for filename in os.listdir(TEMP_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(TEMP_DIR, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    merged_data.append(data)

        with open(f"{input_file}", 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)

        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    except Exception as e:
        print(f"Main error: {str(e)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")


async def check_input_file(input_file):
    """Check if input file exists and contains valid data"""
    if not Path(input_file).exists():
        return False

    async with aiofiles.open(input_file, "r") as f:
        first_line = await f.readline()
        return bool(first_line.strip())


async def prepare_queue(input_file):
    """Prepare processing queue"""
    queue = asyncio.Queue()

    files = os.listdir(TEMP_DIR)

    # Extract UUIDs from existing output files
    ids_in_files = set()
    for file_name in files:
        if file_name.endswith(".json"):
            try:
                file_id = str(file_name[:-5])
                ids_in_files.add(file_id)
            except ValueError:
                continue
    print(ids_in_files)

    async with aiofiles.open(input_file, "r") as f:
        try:
            content = await f.read()
            data_list = json.loads(content)  

            for data in data_list:
                if "id" in data and (data["id"] not in ids_in_files):
                    await queue.put(data["id"])
                    print(f"Added to queue: {data['id']}")
        except json.JSONDecodeError:
            print("Failed to decode JSON.")

    print(f"Queue size: {queue.qsize()}")
    return queue


async def process_item(queue, api_url, api_key, input_file, model_name, progress_bar):
    """Process a single item from queue"""
    async with aiohttp.ClientSession() as session:
        while not queue.empty():
            id = await queue.get()
            #print(f"Processing: {id}")
            await handle_api_request(session, id, api_url, api_key, input_file, model_name, progress_bar)
            queue.task_done()


async def handle_api_request(session, id, api_url, api_key, input_file, model_name, progress_bar):
    """Handle API request and response"""
    original_data = await load_data(input_file,id)
    #print(original_data["instruction"])
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": original_data["instruction"]}],
        "max_tokens": 12000,
        "temperature": 0.6,
    }

    for retry in range(MAX_RETRIES):
        try:
            start_time1 = time.time()

            async with session.post(
                api_url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=INITIAL_TIMEOUT + retry * 10
            ) as response:
                if response.status == 504:
                    raise aiohttp.ClientError("Gateway Timeout")
                response.raise_for_status()

                result = await response.json()
                end_time1 = time.time()
                elapsed_time1 = end_time1 - start_time1
                #print(f"Request time: {elapsed_time1:.2f} seconds")
                await update_data(original_data, result, model_name)
                progress_bar.update(1)
                return

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Attempt {retry + 1}/{MAX_RETRIES} failed: {str(e)}")
            await asyncio.sleep(2 ** retry)

    print(f"Failed to process {id} after all retries.")


async def load_data(input_file,id):
    """Load data"""
    async with aiofiles.open(input_file, "r") as f:
        content = await f.read()
        data_list = json.loads(content)  
        for data in data_list:
            if data.get("id")==id:
                return data
    return None


async def update_data(original, result, model_name):
    """Update original data with response and save"""
    updated = original.copy()
    #print(result)
    if "deepseek-r1" in model_name.lower() or "o3-mini" in model_name.lower():
        updated.update({
            f"{model_name}": result["choices"][0]["message"]["content"],
            f"{model_name}_reasoning": result["choices"][0]["message"]["reasoning_content"],
            f"{model_name}_predict_token": result["usage"]["completion_tokens"],
            f"{model_name}_input_token": result["usage"]["prompt_tokens"],
        })
    else:
        updated.update({
            f"{model_name}": result["choices"][0]["message"]["content"],
            f"{model_name}_predict_token": result["usage"]["completion_tokens"],
            f"{model_name}_input_token": result["usage"]["prompt_tokens"],
        })

    file_name = f"{TEMP_DIR}/{updated['id']}.json"
    #print(file_name)
    await write_to_file(file_name, updated)


async def write_to_file(file_path, data):
    """Write updated data to file"""
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(data) + "\n")
    #print("File written successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", type=str,required=True, help="API endpoint URL")
    parser.add_argument("--api_key", type=str,required=True, help="API key")
    parser.add_argument("--model_name", type=str,required=True, help="Model name")
    parser.add_argument("--dataset_path", type=str,required=True, help="Path to input file")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent workers")
    args = parser.parse_args()

    asyncio.run(main(args.api_url, args.api_key, args.dataset_path, args.concurrency,args.model_name))
