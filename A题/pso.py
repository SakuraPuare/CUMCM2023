# Particle swarm optimization algorithm
import datetime

import matplotlib
import numpy
import pandas as pd
import pyswarms
from matplotlib import pyplot as plt
from pyswarms.utils.plotters import plot_cost_history
from tqdm.contrib.concurrent import process_map

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

from A import (
    Mirror,
    rotate_z,
    compute,
    Sun,
    Tower,
    get_all_time,
)


def arrangement(
    sun,
    tower,
    z,
    w,
    n,
):
    """
    定日镜的排列
    :param sun: 太阳
    :param tower: 吸收塔
    :param z: 定日镜的安装高度
    :param w: 定日镜的尺寸
    :param n: 定日镜的数目

    """
    center = numpy.array([tower.x, tower.y, 0])

    cnt = 0
    lap = 0

    space = 2.5
    h = z if z > w / 2 else w / 2
    r = (w / 2) * numpy.sqrt(2)

    mirror_list = []
    while cnt < n:
        delta = 100 + r + lap * r * 2
        theta = numpy.sin((r + space) / (r + delta))
        v = rotate_z(numpy.array([0, delta + r + lap * r * 2, h]), sun.theta)
        for rad in numpy.linspace(
            (lap % 2) * theta, numpy.pi * 2, int(numpy.pi * 2 // theta)
        ):
            location = center + rotate_z(v, numpy.pi * 2 - rad)
            if numpy.linalg.norm(location[:2]) > 350:
                continue
            mirror_list.append(Mirror(tower, *location, w, w))
            cnt += 1
            if cnt >= n:
                break
        lap += 1

    return mirror_list

    # plt.figure(figsize=(10, 10))
    #
    # # circle
    # theta = numpy.linspace(0, 2 * numpy.pi, 100)
    # x = 350 * numpy.cos(theta)
    # y = 350 * numpy.sin(theta)
    # plt.plot(x, y, color="r")
    # # plt.plot(0.0, 0.0, "O", color="r")
    # plt.xlim(-400, 400)
    # plt.ylim(-400, 400)
    #
    # plt.scatter([i.x for i in mirror_list], [i.y for i in mirror_list])
    # plt.plot(*center[:2], "o", color="b")
    # plt.savefig("full.png")
    # plt.show()
    # return mirror_list


# def compute(args):
#     """
#     :param time: 时间
#     :param r: 吸收塔的 r 半径
#     :param t: 吸收塔的 θ 角度
#     :param z: 定日镜的安装高度
#     :param w: 定日镜的尺寸
#     :param n: 定日镜的数目
#     :return:
#     """
#     (
#         time,
#         r,
#         t,
#         z,
#         w,
#         n,
#     ) = args
#     n = int(n)
#
#     sun = Sun(time)
#     tower = Tower(sun, r * numpy.cos(t), r * numpy.sin(t))
#     mirror_list = arrangement(sun, tower, z, w, n)
#
#     init_mirror_area(tower, mirror_list)
#
#     shadow_efficiency = get_shadow_efficiency(tower, mirror_list)
#     cos_efficiency = get_cosine_efficiency(sun, mirror_list)
#     atmosphere_efficiency = get_atmosphere_efficiency(tower, mirror_list)
#     truncation_efficiency = get_truncation_efficiency(
#         tower, mirror_list, shadow_efficiency
#     )
#     optical_efficiency = (
#         shadow_efficiency
#         * cos_efficiency
#         * atmosphere_efficiency
#         * truncation_efficiency
#         * 0.92
#     )
#
#     output_heat_power = get_output_heat_power(
#         sun, tower, mirror_list, optical_efficiency
#     )
#
#     return numpy.mean(output_heat_power)
#


def function(particles):
    ret = []
    for particle in particles:
        r, t, z, w, n = particle
        x_, y_ = r * numpy.cos(t), r * numpy.sin(t)

        date = get_all_time()

        sun = [Sun(time) for time in date]
        tower = [Tower(s, x_, y_) for s in sun]
        mirror_list = [arrangement(s, t, z, w, n) for s, t in zip(sun, tower)]

        # data = [compute(s, t, m)[-1] for s, t, m in zip(sun, tower, mirror_list)]
        data = [i[-1] for i in process_map(compute, sun, tower, mirror_list)]

        ret.append(60 - numpy.mean(data) / 1e3)

    return numpy.array(ret)


if __name__ == "__main__":
    # sun = Sun(datetime.datetime(2023, 1, 21, 9))
    # tower = Tower(sun, 0, 0)
    # mirror_list = arrangement(sun, tower, 4, 6, 2340, 60)
    # pass

    # 吸收塔的位置坐标、定日镜
    # 尺寸、安装高度、定日镜数目、定日镜位置

    # 吸收塔的位置坐标 定日镜尺寸 定日镜安装高度 定日镜数目
    # , power
    r, t, z, w, n = 268.55513007, 2.57531464, 6.02063777, 5.92792577, 1940

    x_, y_ = r * numpy.cos(t), r * numpy.sin(t)
    x_, y_ = 218.6541654684534, 108.56456125463
    date = get_all_time()
    sun = Sun(date[0])
    tower = Tower(sun, x_, y_)
    mirror_list = arrangement(sun, tower, z, w, n)
    xxx = [i.x for i in mirror_list]
    yyy = [i.y for i in mirror_list]

    pd.DataFrame({"x": xxx, "y": yyy}).to_csv("mirror.csv", index=False)
    pass
    # sun = [Sun(time) for time in date]
    tower = [Tower(s, x_, y_) for s in sun]
    # mirror_list = [ for s, t in zip(sun, tower)]

    compute(sun, tower, mirror_list)

    # lower_bound = [0, 0, 2, 2, 50]
    # upper_bound = [350, numpy.pi * 2, 8, 6, 2000]
    # delta = numpy.array(upper_bound) - numpy.array(lower_bound)
    #
    # options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    #
    # optimizer = pyswarms.single.GlobalBestPSO(
    #     n_particles=8,
    #     dimensions=len(lower_bound),
    #     options=options,
    #     bounds=(lower_bound, upper_bound),
    # )
    #
    # cost, pos = optimizer.optimize(function, iters=50)
    #
    # plot_cost_history(cost_history=optimizer.cost_history)
    # plt.savefig("cost_history.png")
    # plt.show()
