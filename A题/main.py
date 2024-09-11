import datetime
import multiprocessing
import threading

import geopandas
import numpy
import scipy
import tqdm
import pandas
from matplotlib import pyplot as plt
from shapely import Polygon
from tqdm.contrib.concurrent import process_map

phi = numpy.radians(39.4)

fig = plt.figure(figsize=(50, 40))
ax = fig.add_subplot(111, projection="3d")


class Sun:
    def __init__(self, time: datetime.datetime = None):
        if time is None:
            self.time = datetime.datetime.now()
        else:
            self.time = time

        self.omega = numpy.pi / 12 * (self.time.hour + self.time.minute / 60 - 12)
        self.delta = self._delta()
        self.alpha = numpy.arcsin(
            numpy.clip(
                numpy.cos(self.delta) * numpy.cos(phi) * numpy.cos(self.omega)
                + numpy.sin(self.delta) * numpy.sin(phi),
                -1,
                1,
            )
        )
        self.gamma = numpy.arccos(
            numpy.clip(
                (numpy.sin(self.delta) - numpy.sin(self.alpha) * numpy.sin(phi))
                / (numpy.cos(self.alpha) * numpy.cos(phi)),
                -1,
                1,
            )
        )
        self.theta = numpy.pi / 2 - self.alpha
        self.dni = self._dni()
        self.v = self._v()

    def _delta(self):
        """
        太阳赤纬角
        :return: 当前时间下的太阳赤纬角
        """
        d = (self.time - datetime.datetime(self.time.year, 3, 21)).days
        day = (d + 365) % 365
        return numpy.arcsin(
            numpy.clip(
                numpy.sin(2 * numpy.pi * day / 365)
                * numpy.sin(2 * numpy.pi * 23.45 / 360),
                -1,
                1,
            )
        )

    def _dni(self):
        a = 0.34981
        b = 0.5783875
        c = 0.275745
        return 1.366 * (a + b * numpy.exp(-c / numpy.sin(self.alpha)))

    def _v(self):
        return -numpy.array(
            [
                numpy.cos(self.alpha) * numpy.cos(self.gamma),
                numpy.cos(self.alpha) * numpy.sin(self.gamma),
                numpy.sin(self.alpha),
            ]
        )

    def __repr__(self):
        return f"<Sun alpha={self.alpha}, gamma={self.gamma} {self.time}>"


class Tower:
    def __init__(self, sun, x=0, y=0, z=80, h=8, r=3.5, rng=100):
        self.sun = sun

        self.x = x
        self.y = y
        self.z = z

        self.h = h
        self.r = r

        self.range = rng

        self.coordinate = numpy.array([self.x, self.y, self.z])

    def __repr__(self):
        return f"<Tower x={self.x}, y={self.y}, z={self.z}>"


class Mirror:
    def __init__(self, tower, sun, x, y, z=4, h=6, w=6):
        self.tower = tower
        self.sun = sun

        self.x = x
        self.y = y
        self.z = z

        self.h = h
        self.w = w

        self.area = 0

        self.coordinate = numpy.array([self.x, self.y, self.z])
        self.to_tower = self.tower.coordinate - self.coordinate
        self.n = self._n()

    def _n(self):
        v = self.sun.n + self.to_tower
        return v / numpy.linalg.norm(v)

    def points(self):
        x = numpy.cross(self.n, [0, 0, 1])
        y = numpy.cross(self.n, x)

        left_down = self.coordinate - 0.5 * self.h * y - 0.5 * self.w * x
        left_up = self.coordinate - 0.5 * self.h * y + 0.5 * self.w * x
        right_up = self.coordinate + 0.5 * self.h * y + 0.5 * self.w * x
        right_down = self.coordinate + 0.5 * self.h * y - 0.5 * self.w * x

        return numpy.array([left_down, left_up, right_up, right_down])

    def project(self):
        ans = numpy.array(
            [
                point + (-point[2] / self.sun.n[2] * self.sun.n)
                for point in self.points()
            ]
        )
        # z = 0
        ans[:, 2] = 0
        return ans

    def project_polygon(self):
        return Polygon(self.project())

    def __gt__(self, other):
        return abs(self.x) + abs(self.y) > abs(other.x) + abs(other.y)

    def __repr__(self):
        return f"<Mirror x={self.x}, y={self.y}>"


def init_mirror_area(mirrors):
    poly = [mirror.project_polygon() for mirror in mirrors]

    data = []
    last = 0
    for i in range(1, len(poly) + 1):
        a = geopandas.GeoDataFrame(geometry=poly[:i]).area.sum() - last
        last += a
        mirrors[i - 1].area = a


def get_shading_efficiency(mirrors):
    """
    阴影遮挡效率
    :param mirrors: 镜面列表
    :return: 阴影遮挡效率
    """
    geo = [i.project_polygon() for i in mirrors]

    gdf = geopandas.GeoDataFrame(geometry=geo)

    area = gdf.area.sum()

    intersection = gdf.unary_union.intersection(gdf.unary_union)
    overlap = intersection.area
    # 重复部分
    return numpy.array([overlap / area] * len(mirrors))


def get_cosine_efficiency(mirrors):
    """
    余弦效率
    :param mirrors: 镜面列表
    :return: 余弦效率
    """

    # 计算 mirror.sun.v 与 mirror.n 的夹角余弦
    return numpy.array([numpy.dot(-mirror.sun.n, mirror.n) for mirror in mirrors])


def get_atmospheric_efficiency(mirrors):
    """
    大气效率
    :param mirrors: 镜面列表
    :return: 大气效率
    """
    return numpy.array(
        [
            0.99321 - 0.0001176 * d + 1.97 * 1e-8 * numpy.power(d, 2)
            for d in [numpy.linalg.norm(mirror.to_tower) for mirror in mirrors]
        ]
    )


def get_truncation_efficiency(mirror):
    """
    截断效率
    :param mirror: 镜面
    :return: 截断效率
    """
    # 二维正态分布
    x_lower = mirror.tower.x - mirror.tower.r / 2
    x_upper = mirror.tower.x + mirror.tower.r / 2
    y = mirror.tower.h * (
        1 / numpy.cos(numpy.arctan(mirror.n[2] / numpy.linalg.norm(mirror.n[:2])))
    )
    y_lower = mirror.tower.y - y / 2
    y_upper = mirror.tower.y + y / 2

    def pdf(x, y):
        v = numpy.array([x, y])
        sigmal = numpy.linalg.norm(mirror.to_tower) * 0.005
        return (
            1
            / (2 * numpy.pi * sigmal**2)
            * numpy.exp(-numpy.linalg.norm(v) ** 2 / (2 * sigmal**2))
        )

    return scipy.integrate.dblquad(pdf, x_lower, x_upper, y_lower, y_upper)[0]


def get_heat_output_power(mirrors, optical_efficiency):
    """
    镜面输出功率
    :param mirrors: 镜面列表
    :return: 镜面输出功率
    """
    return numpy.array([36 * i.sun.dni for i in mirrors] * optical_efficiency)


def main():
    pd = pandas.DataFrame(
        columns=["日期", "平均光学效率", "平均余弦效率", "平均遮挡效率", "平均截断效率", "单位面积镜面年平均输出热功率"]
    )

    day_list = [datetime.datetime(2023, month, 21) for month in range(1, 13)]
    time_list = [
        [
            datetime.datetime(day.year, day.month, day.day, hour, minute)
            for hour, minute in [(9, 0), (10, 30), (12, 0), (13, 30), (15, 0)]
        ]
        for day in day_list
    ]

    for day in time_list:
        sun = [Sun(time) for time in day]
        tower = [Tower(sun) for sun in sun]
        mirrors = [
            sorted([Mirror(tower, sun, x[i], y[i]) for i in range(len(x))])
            for tower, sun in zip(tower, sun)
        ]

        [init_mirror_area(i) for i in mirrors]

        shading_efficiency_list = numpy.clip(
            numpy.array(process_map(get_shading_efficiency, mirrors)), 0, 1
        )

        cosine_efficiency_list = numpy.clip(
            numpy.array(process_map(get_cosine_efficiency, mirrors)), 0, 1
        )

        atmospheric_efficiency_list = numpy.clip(
            numpy.array(process_map(get_atmospheric_efficiency, mirrors)), 0, 1
        )

        truncation_efficiency_list = numpy.clip(
            numpy.array(
                [
                    numpy.array(
                        process_map(get_truncation_efficiency, mirror, chunksize=1)
                    )
                    for mirror in mirrors
                ]
            ),
            0,
            1,
        )

        optical_efficiency_list = (
            shading_efficiency_list
            * cosine_efficiency_list
            * atmospheric_efficiency_list
            * truncation_efficiency_list
            * 0.92
        )

        output_power_list = numpy.array(
            process_map(
                get_heat_output_power,
                mirrors,
                optical_efficiency_list,
            )
        )

        # pd 插入一行
        data = numpy.array(
            [
                day_list[len(pd) - 1],
                numpy.mean(optical_efficiency_list),
                numpy.mean(cosine_efficiency_list),
                numpy.mean(shading_efficiency_list),
                numpy.mean(truncation_efficiency_list),
                numpy.sum(output_power_list) / (6 * 6 * len(output_power_list)),
            ]
        )

        pd.loc[len(pd)] = data
        pd.to_excel("char1.xlsx")


#
# def
#
# def question_one():
#     day_list = [datetime.datetime(2023, month, 21) for month in range(1, 13)]
#     time_list = numpy.array(
#         [
#             [
#                 datetime.datetime(day.year, day.month, day.day, hour, minute)
#                 for hour, minute in [(9, 0), (10, 30), (12, 0), (13, 30), (15, 0)]
#             ]
#             for day in day_list
#         ]
#     )
#
#     numpy.reshape(time_list, (-1, 5))
#
#     mirrors = [get_mirrors_stats(time) for time in time_list]
#
#     pd = pandas.DataFrame(columns=["日期", "平均光学效率", "平均余弦效率", "平均遮挡效率", "平均截断效率"])
#
#     date_name = [day.strftime("%m月%d日") for day in day_list]
#
#     avg_shading_efficiency_list = numpy.clip(
#         numpy.array(process_map(get_shading_efficiency, mirrors)), 0, 1
#     )
#     avg_cosine_efficiency_list = numpy.clip(
#         numpy.array(process_map(get_cosine_efficiency, mirrors)), 0, 1
#     )
#     avg_atmospheric_efficiency_list = numpy.clip(
#         numpy.array(process_map(get_atmospheric_efficiency, mirrors), 0, 1)
#     )
#
#     avg_truncation_efficiency_list = numpy.clip(
#         numpy.array(process_map(get_truncation_efficiency, mirrors), 0, 1)
#     )
#
#     # [get_truncation_efficiency(mirror) for mirror in mirrors]
#
#     avg_optical_efficiency_list = (
#         avg_shading_efficiency_list
#         * avg_cosine_efficiency_list
#         * avg_atmospheric_efficiency_list
#         * avg_truncation_efficiency_list
#         * 0.92
#     )
#
#     mirror_output_power_list = numpy.array(
#         process_map(
#             get_mirror_output_power,
#             time_list,
#             avg_optical_efficiency_list,
#         )
#     )
#     # mirror_output_power_list = numpy.array(
#     #     [
#     #         get_mirror_output_power(time, avg_optical_efficiency_list)
#     #         for time in time_list
#     #     ]
#     # )
#
#     pd["日期"] = date_name
#     pd["平均光学效率"] = avg_optical_efficiency_list
#     pd["平均余弦效率"] = avg_cosine_efficiency_list
#     pd["平均遮挡效率"] = avg_shading_efficiency_list
#     pd["平均截断效率"] = avg_truncation_efficiency_list
#     pd["单位面积镜面平均输出功率"] = mirror_output_power_list / (6 * 6 * len(mirrors))
#
#     pd.to_excel("char1.xlsx")
#
#     pd = pandas.DataFrame(
#         columns=[
#             "年平均光学效率",
#             "年平均余弦效率",
#             "年平均阴影遮挡效率",
#             "年平均截断效率",
#             "年平均输出热功率",
#             "单位面积镜面年平均输出热功率",
#         ]
#     )
#     pd["年平均光学效率"] = numpy.mean(avg_optical_efficiency_list)
#     pd["年平均余弦效率"] = numpy.mean(avg_cosine_efficiency_list)
#     pd["年平均阴影遮挡效率"] = numpy.mean(avg_shading_efficiency_list)
#     pd["年平均截断效率"] = numpy.mean(avg_truncation_efficiency_list)
#     pd["年平均输出热功率"] = numpy.mean(mirror_output_power_list)
#     pd["单位面积镜面年平均输出热功率"] = numpy.mean(mirror_output_power_list) / (6 * 6 * len(mirrors))
#
#     pd.to_excel("char2.xlsx")
#
#
# def get_year_average_optical_efficiency():
#     day_list = [datetime.datetime(2023, month, 21) for month in range(1, 13)]
#     time_list = []
#     for day in day_list:
#         time_list.extend(
#             [
#                 datetime.datetime(day.year, day.month, day.day, hour, minute)
#                 for hour, minute in [(9, 0), (10, 30), (12, 0), (13, 30), (15, 0)]
#             ]
#         )
#
#     for time in time_list:
#         sun = Sun(time)
#         tower = Tower(sun)
#         mirrors = sorted([Mirror(tower, sun, x[i], y[i]) for i in range(len(x))])
#         geo = [i.project()_polygon for i in mirrors]
#
#         gdf = geopandas.GeoDataFrame(geometry=geo)
#
#         area = gdf.area.sum()
#
#         intersection = gdf.unary_union.intersection(gdf.unary_union)
#         overlap = intersection.area
#
#         pass
#
#         # for mirror in tqdm.tqdm(mirrors):
#         #     ax.plot(
#         #         *mirror.coordinate.T,
#         #         "o",
#         #         color="red",
#         #     )
#
#         # for point in mirror.points():
#         #     ax.plot(
#         #         *point.T,
#         #         "o",
#         #         color="blue",
#         #     )
#
#         prj = [i.project() for i in mirrors]
#         for m in tqdm.tqdm(mirrors):
#             pro = m.project()
#             for j in range(len(pro)):
#                 coor = numpy.array([pro[j], pro[(j + 1) % len(pro)]])
#                 ax.plot(coor.T[0], coor.T[1], coor.T[2], color="red", label="Line")
#
#         ax.plot(*numpy.stack((numpy.zeros((3,)), tower.coordinate)).T, color="r")
#
#         # view from top
#         ax.view_init(elev=0, azim=0)  # 仰角elev为90
#
#         # 缩放z
#         plt.savefig("1.png")
#         plt.show()
#         pass


if __name__ == "__main__":
    data = pandas.read_excel("附件.xlsx")
    x = data["x坐标 (m)"]
    y = data["y坐标 (m)"]

    main()

    # question_one()
    # get_year_average_optical_efficiency()

    pass
