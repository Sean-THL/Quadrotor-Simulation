import time
import csv
import numpy as np

def path_length(path):
    if path is None or len(path) < 2:
        return None
    p = np.array(path, dtype=float)
    return float(np.sum(np.linalg.norm(p[1:] - p[:-1], axis=1)))

def plan_once(seed: int, max_iter=500, goal_sample_rate=0.1):
    np.random.seed(seed)

    # TODO: 依你在 PathPlanning 看到的 API 替換：
    from PathPlanning import Map RRT, RRTStar
    bounds=np.array([0, 100])
    mapobs=Map(obstacle_list, bounds, path_resolution, dim=3)
    start = np.array([...]); goal = np.array([...])

    # from PathPlanning.XXX import Map, RRTStar
    # bounds = np.array([0,100])
    # mapobs = Map(obstacles, bounds, dim=3)
    # start = np.array([...]); goal = np.array([...])
    # rrt = RRTStar(start=start, goal=goal, Map=mapobs, max_iter=max_iter, goal_sample_rate=goal_sample_rate)
    # waypoints, min_cost = rrt.plan()

    waypoints, min_cost = None, None  # placeholder
    success = waypoints is not None and len(waypoints) > 1
    return waypoints, min_cost, success

def main():
    seeds = list(range(20))
    rows = []

    for s in seeds:
        t0 = time.perf_counter()
        waypoints, cost, ok = plan_once(s)
        dt = time.perf_counter() - t0
        plen = path_length(waypoints)
        rows.append([s, ok, dt, cost, plen])

    succ = [r for r in rows if r[1]]
    print(f"Runs={len(rows)}  Success={len(succ)}/{len(rows)} ({len(succ)/len(rows):.2f})")
    if succ:
        print("Avg time(s):", sum(r[2] for r in succ)/len(succ))

    with open("planning_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed","success","time_s","cost","path_len"])
        w.writerows(rows)

if __name__ == "__main__":
    main()
