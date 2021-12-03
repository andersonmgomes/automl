#https://realpython.com/python-concurrency/
import multiprocessing
import time

soma = 0
def cpu_bound(number):
    return sum(i * i for i in range(number))


def find_sums(numbers):
    with multiprocessing.Pool() as pool:
        return pool.map(cpu_bound, numbers)


if __name__ == "__main__":
    numbers = [x for x in range(20)]
    print(soma)
    start_time = time.time()
    print(find_sums(numbers))
    duration = time.time() - start_time
    print(soma)
    print(f"Duration {duration} seconds")