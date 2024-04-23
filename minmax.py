import numpy as np
import concurrent.futures
import multiprocessing
from functools import reduce
from multiprocessing import freeze_support

def reduce_chunk(chunk, operation):
    if operation == 'min':
        return np.min(chunk)
    elif operation == 'max':
        return np.max(chunk)
    elif operation == 'sum':
        return np.sum(chunk)
    elif operation == 'average':
        return np.mean(chunk)

def parallel_reduction(data, operation):
    num_processes = min(len(data), multiprocessing.cpu_count())
    chunk_size = len(data) // num_processes

    # Split the data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Perform reduction operation in parallel
        results = list(executor.map(reduce_chunk, chunks, [operation]*len(chunks)))

    # Combine results
    if operation in ['min', 'max', 'sum']:
        if operation == 'min':
            return reduce(lambda x, y: x if x <= y else y, results)
        elif operation == 'max':
            return reduce(lambda x, y: x if x >= y else y, results)
        elif operation == 'sum':
            return sum(results)
    elif operation == 'average':
        total_sum = sum(results)
        return total_sum / len(data)

if __name__ == '__main__':
    # Example data
    data = np.random.randint(10, 100, size=100)
    print(data)

    # Perform parallel reduction operations
    min_value = parallel_reduction(data, 'min')
    max_value = parallel_reduction(data, 'max')
    sum_value = parallel_reduction(data, 'sum')
    average_value = parallel_reduction(data, 'average')

    print("Min:", min_value)
    print("Max:", max_value)
    print("Sum:", sum_value)
    print("Average:", average_value)
    freeze_support()
