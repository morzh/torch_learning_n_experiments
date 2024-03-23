import time


def execution_time(fn):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        fn(*args, **kwargs)
        end_time = time.time()
        print('execution time is:', end_time - start_time, 'seconds')

    return wrapper


if __name__ == '__main__':
    @execution_time
    def test_fn():
        time.sleep(3)


    test_fn()
