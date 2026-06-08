import os
import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


class RequestStats:
    def __init__(self):
        self.total_requests = 0
        self.completed_requests = 0
        self.rejected_requests = 0

        self.waiting_times = []
        self.response_times = []
        self.queue_lengths = []

        self.busy_time = 0.0


def handle_request(env, request_id, server, stats, mean_service_time, arrival_time):
    """
    Обработка одного HTTP-запроса.
    """

    with server.request() as req:
        yield req

        start_service_time = env.now
        waiting_time = start_service_time - arrival_time

        service_time = random.expovariate(1.0 / mean_service_time)
        yield env.timeout(service_time)

        finish_time = env.now
        response_time = finish_time - arrival_time

        stats.completed_requests += 1
        stats.waiting_times.append(waiting_time)
        stats.response_times.append(response_time)
        stats.busy_time += service_time


def request_generator(
    env,
    server,
    stats,
    arrival_rate,
    mean_service_time,
    queue_limit,
    simulation_time
):
    """
    Генератор HTTP-запросов.
    """

    request_id = 0

    while env.now < simulation_time:
        interarrival_time = random.expovariate(arrival_rate)
        yield env.timeout(interarrival_time)

        request_id += 1
        stats.total_requests += 1

        current_queue_length = len(server.queue)
        stats.queue_lengths.append(current_queue_length)

        # Если все обработчики заняты и очередь заполнена — отказ
        if server.count == server.capacity and len(server.queue) >= queue_limit:
            stats.rejected_requests += 1
            continue

        arrival_time = env.now

        env.process(
            handle_request(
                env,
                request_id,
                server,
                stats,
                mean_service_time,
                arrival_time
            )
        )


def run_simulation(
    seed=1,
    arrival_rate=20,
    mean_service_time=0.05,
    workers=2,
    queue_limit=100,
    simulation_time=100
):
    """
    Запуск одного эксперимента.
    """

    random.seed(seed)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=workers)
    stats = RequestStats()

    env.process(
        request_generator(
            env,
            server,
            stats,
            arrival_rate,
            mean_service_time,
            queue_limit,
            simulation_time
        )
    )

    env.run(until=simulation_time)

    if stats.completed_requests > 0:
        avg_waiting_time = np.mean(stats.waiting_times)
        avg_response_time = np.mean(stats.response_times)
        p95_response_time = np.percentile(stats.response_times, 95)
    else:
        avg_waiting_time = 0
        avg_response_time = 0
        p95_response_time = 0

    rejection_probability = (
        stats.rejected_requests / stats.total_requests
        if stats.total_requests > 0
        else 0
    )

    utilization = stats.busy_time / (workers * simulation_time)

    return {
        "seed": seed,
        "arrival_rate": arrival_rate,
        "mean_service_time": mean_service_time,
        "workers": workers,
        "queue_limit": queue_limit,
        "simulation_time": simulation_time,
        "total_requests": stats.total_requests,
        "completed_requests": stats.completed_requests,
        "rejected_requests": stats.rejected_requests,
        "rejection_probability": rejection_probability,
        "avg_waiting_time": avg_waiting_time,
        "avg_response_time": avg_response_time,
        "p95_response_time": p95_response_time,
        "max_queue_length": max(stats.queue_lengths) if stats.queue_lengths else 0,
        "avg_queue_length": np.mean(stats.queue_lengths) if stats.queue_lengths else 0,
        "utilization": utilization,
    }


def confidence_interval_95(values):
    """
    Расчёт 95% доверительного интервала для среднего значения.
    Возвращает половину ширины интервала.
    """

    values = np.array(values)

    if len(values) < 2:
        return 0

    # Если все значения одинаковые, разброса нет
    if np.std(values, ddof=1) == 0:
        return 0

    standard_error = scipy_stats.sem(values)

    interval = scipy_stats.t.interval(
        confidence=0.95,
        df=len(values) - 1,
        loc=np.mean(values),
        scale=standard_error
    )

    half_width = (interval[1] - interval[0]) / 2
    return half_width


def run_repeated_experiments(
    arrival_rate,
    workers,
    queue_limit,
    mean_service_time=0.05,
    simulation_time=100,
    runs=20
):
    """
    Несколько прогонов одной конфигурации с разными seed.
    """

    results = []

    for seed in range(1, runs + 1):
        result = run_simulation(
            seed=seed,
            arrival_rate=arrival_rate,
            mean_service_time=mean_service_time,
            workers=workers,
            queue_limit=queue_limit,
            simulation_time=simulation_time
        )

        results.append(result)

    return pd.DataFrame(results)


def summarize_experiment(df):
    """
    Усреднение результатов серии прогонов.
    """

    summary = {
        "arrival_rate": df["arrival_rate"].iloc[0],
        "workers": df["workers"].iloc[0],
        "queue_limit": df["queue_limit"].iloc[0],

        "avg_response_time_mean": df["avg_response_time"].mean(),
        "avg_response_time_ci95": confidence_interval_95(df["avg_response_time"]),

        "p95_response_time_mean": df["p95_response_time"].mean(),
        "p95_response_time_ci95": confidence_interval_95(df["p95_response_time"]),

        "rejection_probability_mean": df["rejection_probability"].mean(),
        "rejection_probability_ci95": confidence_interval_95(df["rejection_probability"]),

        "avg_queue_length_mean": df["avg_queue_length"].mean(),
        "max_queue_length_mean": df["max_queue_length"].mean(),

        "utilization_mean": df["utilization"].mean(),
    }

    return summary


def experiment_by_arrival_rate():
    """
    Эксперимент 1:
    исследуем влияние интенсивности входящих запросов.
    """

    arrival_rates = [10, 20, 30, 35, 40, 45, 50, 60]
    all_summaries = []

    for arrival_rate in arrival_rates:
        df = run_repeated_experiments(
            arrival_rate=arrival_rate,
            workers=2,
            queue_limit=100,
            mean_service_time=0.012,
            simulation_time=100,
            runs=20
        )

        summary = summarize_experiment(df)
        all_summaries.append(summary)

    result_df = pd.DataFrame(all_summaries)

    result_df.to_csv(
        "results/experiment_arrival_rate.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return result_df

def experiment_by_workers():
    """
    Эксперимент 2:
    исследуем влияние количества обработчиков сервера.
    """

    workers_list = [1, 2, 3, 4, 5, 6]
    all_summaries = []

    for workers in workers_list:
        df = run_repeated_experiments(
            arrival_rate=50,
            workers=workers,
            queue_limit=100,
            mean_service_time=0.05,
            simulation_time=100,
            runs=20
        )

        summary = summarize_experiment(df)
        all_summaries.append(summary)

    result_df = pd.DataFrame(all_summaries)

    result_df.to_csv(
        "results/experiment_workers.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return result_df

def experiment_by_queue_limit():
    """
    Эксперимент 3:
    исследуем влияние размера очереди.
    """

    queue_limits = [0, 10, 25, 50, 100, 200]
    all_summaries = []

    for queue_limit in queue_limits:
        df = run_repeated_experiments(
            arrival_rate=45,
            workers=2,
            queue_limit=queue_limit,
            mean_service_time=0.05,
            simulation_time=100,
            runs=20
        )

        summary = summarize_experiment(df)
        all_summaries.append(summary)

    result_df = pd.DataFrame(all_summaries)

    result_df.to_csv(
        "results/experiment_queue_limit.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return result_df

def build_queue_plots(df):
    """
    Построение графиков по эксперименту с размером очереди.
    """

    plt.figure()
    plt.plot(df["queue_limit"], df["avg_response_time_mean"], marker="o")
    plt.xlabel("Размер очереди")
    plt.ylabel("Среднее время ответа, сек")
    plt.title("Влияние размера очереди на среднее время ответа")
    plt.grid(True)
    plt.savefig("results/queue_avg_response_time.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(df["queue_limit"], df["p95_response_time_mean"], marker="o")
    plt.xlabel("Размер очереди")
    plt.ylabel("95-й перцентиль времени ответа, сек")
    plt.title("Влияние размера очереди на 95-й перцентиль")
    plt.grid(True)
    plt.savefig("results/queue_p95_response_time.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(df["queue_limit"], df["rejection_probability_mean"], marker="o")
    plt.xlabel("Размер очереди")
    plt.ylabel("Вероятность отказа")
    plt.title("Влияние размера очереди на вероятность отказа")
    plt.grid(True)
    plt.savefig("results/queue_rejection_probability.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(df["queue_limit"], df["avg_queue_length_mean"], marker="o")
    plt.xlabel("Размер очереди")
    plt.ylabel("Средняя длина очереди")
    plt.title("Влияние размера очереди на среднюю длину очереди")
    plt.grid(True)
    plt.savefig("results/queue_avg_queue_length.png", dpi=300, bbox_inches="tight")

    plt.close("all")

def build_worker_plots(df):
    """
    Построение графиков по эксперименту с количеством обработчиков.
    """

    plt.figure()
    plt.plot(df["workers"], df["avg_response_time_mean"], marker="o")
    plt.xlabel("Количество обработчиков")
    plt.ylabel("Среднее время ответа, сек")
    plt.title("Влияние количества обработчиков на среднее время ответа")
    plt.grid(True)
    plt.savefig("results/workers_avg_response_time.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(df["workers"], df["p95_response_time_mean"], marker="o")
    plt.xlabel("Количество обработчиков")
    plt.ylabel("95-й перцентиль времени ответа, сек")
    plt.title("Влияние количества обработчиков на 95-й перцентиль")
    plt.grid(True)
    plt.savefig("results/workers_p95_response_time.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(df["workers"], df["rejection_probability_mean"], marker="o")
    plt.xlabel("Количество обработчиков")
    plt.ylabel("Вероятность отказа")
    plt.title("Влияние количества обработчиков на вероятность отказа")
    plt.grid(True)
    plt.savefig("results/workers_rejection_probability.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(df["workers"], df["utilization_mean"], marker="o")
    plt.xlabel("Количество обработчиков")
    plt.ylabel("Загрузка обработчиков")
    plt.title("Влияние количества обработчиков на загрузку")
    plt.grid(True)
    plt.savefig("results/workers_utilization.png", dpi=300, bbox_inches="tight")

    plt.close("all")


def build_plots(df):
    """
    Построение графиков по результатам эксперимента.
    """

    plt.figure()
    plt.plot(df["arrival_rate"], df["avg_response_time_mean"], marker="o")
    plt.xlabel("Интенсивность запросов, λ (запросов/сек)")
    plt.ylabel("Среднее время ответа, сек")
    plt.title("Зависимость среднего времени ответа от нагрузки")
    plt.grid(True)
    plt.savefig("results/avg_response_time.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(df["arrival_rate"], df["p95_response_time_mean"], marker="o")
    plt.xlabel("Интенсивность запросов, λ (запросов/сек)")
    plt.ylabel("95-й перцентиль времени ответа, сек")
    plt.title("Зависимость 95-го перцентиля времени ответа от нагрузки")
    plt.grid(True)
    plt.savefig("results/p95_response_time.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(df["arrival_rate"], df["rejection_probability_mean"], marker="o")
    plt.xlabel("Интенсивность запросов, λ (запросов/сек)")
    plt.ylabel("Вероятность отказа")
    plt.title("Зависимость вероятности отказа от нагрузки")
    plt.grid(True)
    plt.savefig("results/rejection_probability.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(df["arrival_rate"], df["utilization_mean"], marker="o")
    plt.xlabel("Интенсивность запросов, λ (запросов/сек)")
    plt.ylabel("Загрузка обработчиков")
    plt.title("Зависимость загрузки обработчиков от нагрузки")
    plt.grid(True)
    plt.savefig("results/utilization.png", dpi=300, bbox_inches="tight")

    plt.close("all")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Эксперимент 1: влияние интенсивности входящего потока")
    arrival_rate_df = experiment_by_arrival_rate()
    print(arrival_rate_df.to_string(index=False))
    build_plots(arrival_rate_df)

    print("\nЭксперимент 2: влияние количества обработчиков")
    workers_df = experiment_by_workers()
    print(workers_df.to_string(index=False))
    build_worker_plots(workers_df)

    print("\nЭксперимент 3: влияние размера очереди")
    queue_df = experiment_by_queue_limit()
    print(queue_df.to_string(index=False))
    build_queue_plots(queue_df)

    print("\nФайлы сохранены в папку results/")