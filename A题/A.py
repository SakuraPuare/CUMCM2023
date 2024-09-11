import datetime

import geopandas
import matplotlib
import numpy
import pandas
import tqdm
from matplotlib import pyplot as plt
from shapely import Polygon
from tqdm.contrib.concurrent import process_map

phi = numpy.radians(39.4)

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def rotate_x(v, theta):
    """
    绕x轴旋转
    :param v:
    :param theta:
    :return:
    """
    return numpy.array(
        [
            v[0],
            v[1] * numpy.cos(theta) - v[2] * numpy.sin(theta),
            v[1] * numpy.sin(theta) + v[2] * numpy.cos(theta),
        ]
    )


def rotate_y(v, theta):
    """
    绕y轴旋转
    :param v:
    :param theta:
    :return:
    """
    return numpy.array(
        [
            v[0] * numpy.cos(theta) + v[2] * numpy.sin(theta),
            v[1],
            -v[0] * numpy.sin(theta) + v[2] * numpy.cos(theta),
        ]
    )


def rotate_z(v, theta):
    """
    绕z轴旋转
    :param v:
    :param theta:
    :return:
    """
    return numpy.array(
        [
            v[0] * numpy.cos(theta) - v[1] * numpy.sin(theta),
            v[0] * numpy.sin(theta) + v[1] * numpy.cos(theta),
            v[2],
        ]
    )


class Sun:
    G = 1.366  # 太阳常数

    def __init__(self, time):
        self.time = time

        self.delta = self._delta()
        self.omega = self._omega()
        self.alpha = self._alpha()
        self.gamma = self._gamma()
        self.theta = self._theta()
        self.dni = self._dni()

        self.n = self._n()

    def __repr__(self):
        return f"<Sun time={self.time}>"

    def _alpha(self):
        return numpy.arcsin(
            numpy.clip(
                numpy.cos(self.delta) * numpy.cos(phi) * numpy.cos(self.omega)
                + numpy.sin(self.delta) * numpy.sin(phi),
                -1,
                1,
            )
        )
        pass

    def _gamma(self):
        return numpy.arccos(
            numpy.clip(
                (numpy.sin(self.delta) - numpy.sin(self.alpha) * numpy.sin(phi))
                / (numpy.cos(self.alpha) * numpy.cos(phi)),
                -1,
                1,
            )
        )

    def _omega(self):
        return numpy.pi / 12 * (self.time.hour + self.time.minute / 60 - 12)

    def _delta(self):
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

    def _theta(self):
        return numpy.pi / 2 - self.alpha

    def _dni(self):
        a = 0.34981
        b = 0.5783875
        c = 0.275745
        return self.G * (a + b * numpy.exp(-c / numpy.sin(self.alpha)))

    def _n(self):
        return -numpy.array(
            [
                numpy.cos(self.alpha) * numpy.cos(self.gamma),
                numpy.cos(self.alpha) * numpy.sin(self.gamma),
                numpy.sin(self.alpha),
            ]
        )


class Tower:
    def __init__(self, sun: Sun, x=0, y=0, z=80, h=8, r=3.5):
        self.sun = sun

        self.x = x
        self.y = y
        self.z = z

        self.coordinate = numpy.array([x, y, z])

        self.h = h
        self.r = r

        self.center = numpy.array([x, y, 0])

        self.vertex = self._vertex()
        self.project = self._project()

    @property
    def vertex_area(self):
        return Polygon(self.vertex).area

    def _vertex(self):
        # 变换后的宽高
        h = (self.z + self.h / 2) / numpy.cos(self.sun.alpha)
        w = self.r * 2

        x = [1, -self.sun.n[0] / self.sun.n[1], 0]
        x = x / numpy.linalg.norm(x)
        y = numpy.cross(self.sun.n, x)

        # 底边中心 为原点
        left_down = -0.5 * w * x
        left_up = 0.5 * w * x
        right_up = 1 * h * y + 0.5 * w * x
        right_down = 1 * h * y - 0.5 * w * x

        return numpy.array([left_down, left_up, right_up, right_down])

    def _project(self):
        ans = numpy.array(
            [point + (-point[2] / self.sun.n[2] * self.sun.n) for point in self.vertex]
        )
        # z = 0
        ans[:, 2] = 0
        return ans

    def __repr__(self):
        return f"<Tower x={self.x}, y={self.y}, z={self.z}>"


class Mirror:
    def __init__(
        self,
        tower: Tower,
        x: float = 0,
        y: float = 0,
        z: float = 4,
        h: float = 6,
        w: float = 6,
    ):
        self.tower = tower

        self.x = x
        self.y = y
        self.z = z

        self.coordinate = numpy.array([x, y, z])

        self.h = h
        self.w = w
        self.r = numpy.sqrt(h**2 + w**2) / 2

        self.v_tower = self._v_tower()
        self.n_tower = self._n_tower()

        self.n = self._n()  # 平面法向量

        self.vertex = self._vertex()
        self.project = self._project()

        self.area = 0

    def _v_tower(self):
        return self.tower.coordinate - self.coordinate

    def _n_tower(self):
        return self.v_tower / numpy.linalg.norm(self.v_tower)

    def _n(self):
        v = self.n_tower - self.tower.sun.n
        return v / numpy.linalg.norm(v)

    def _vertex(self):
        x = [1, -self.n[0] / self.n[1], 0]
        x = x / numpy.linalg.norm(x)
        y = numpy.cross(self.n, x)

        left_down = self.coordinate - 0.5 * self.h * y - 0.5 * self.w * x
        left_up = self.coordinate - 0.5 * self.h * y + 0.5 * self.w * x
        right_up = self.coordinate + 0.5 * self.h * y + 0.5 * self.w * x
        right_down = self.coordinate + 0.5 * self.h * y - 0.5 * self.w * x

        return numpy.array([left_down, left_up, right_up, right_down])

    def _project(self):
        ans = numpy.array(
            [
                point + (-point[2] / self.tower.sun.n[2] * self.tower.sun.n)
                for point in self.vertex
            ]
        )
        # z = 0
        ans[:, 2] = 0
        return ans

    def __repr__(self):
        return f"<Mirror x={self.x}, y={self.y}, z={self.z}>"


def get_shadow_efficiency(tower, mirror_list):
    geo = [Polygon(i.project) for i in mirror_list]
    geo.append(Polygon(tower.project))

    gdf = geopandas.GeoDataFrame(geometry=geo)

    area = gdf.area.sum()

    intersection = gdf.unary_union.intersection(gdf.unary_union)
    overlap = intersection.area

    # 重复部分
    return numpy.array([overlap / area] * len(mirror_list))


def get_cosine_efficiency(sun, mirror_list):
    return numpy.clip(numpy.cos(sun.delta), -1, 1).repeat(len(mirror_list))


def get_atmosphere_efficiency(tower: object, mirror_list: object) -> numpy.array:
    coordinate = tower.coordinate
    distance = numpy.array(
        [numpy.linalg.norm(mirror.coordinate - coordinate) for mirror in mirror_list]
    )
    return numpy.array(
        [0.99321 - 0.0001176 * d + 1.97 * 1e-8 * numpy.power(d, 2) for d in distance]
    )


def get_truncation_efficiency(tower, mirror_list, shadow_efficiency):
    dat = []
    for index, mirror in enumerate(mirror_list):
        dis = mirror.h / numpy.sin(4.65 * 10**-3)
        d = numpy.linalg.norm(mirror.coordinate[:2] - tower.coordinate[:2])
        ratio = (dis + d) / dis

        mirror_area = mirror.w * mirror.w * ratio
        theta = mirror.v_tower[2] / numpy.linalg.norm(mirror.v_tower[:2])

        head_area = tower.r * 2 * tower.h * numpy.cos(theta)
        dat.append(mirror_area / head_area * shadow_efficiency[index])
    return numpy.array(dat)


def get_output_heat_power(sun, tower, mirror_list, optical_efficiency):
    return sun.dni * numpy.sum(
        [mirror.area * optical_efficiency[i] for i, mirror in enumerate(mirror_list)]
    )


def compute(sun, tower, mirror_list):
    # init_mirror_area(tower, mirror_list)

    shadow_efficiency = get_shadow_efficiency(tower, mirror_list)
    cos_efficiency = get_cosine_efficiency(sun, mirror_list)
    atmosphere_efficiency = get_atmosphere_efficiency(tower, mirror_list)
    truncation_efficiency = get_truncation_efficiency(
        tower, mirror_list, shadow_efficiency
    )
    optical_efficiency = (
        shadow_efficiency
        * cos_efficiency
        * atmosphere_efficiency
        * truncation_efficiency
        * 0.92
    )

    output_heat_power = get_output_heat_power(
        sun, tower, mirror_list, optical_efficiency
    )

    return numpy.array(
        [
            sun.time,
            numpy.mean(optical_efficiency),
            numpy.mean(cos_efficiency),
            numpy.mean(shadow_efficiency),
            numpy.mean(truncation_efficiency),
            output_heat_power,
        ]
    )


def init_mirror_area(tower, mirrors):
    poly = [Polygon(mirror.project) for mirror in mirrors]
    poly.append(Polygon(tower.project))

    last = 0
    for i in range(1, len(poly)):
        a = geopandas.GeoDataFrame(geometry=poly[:i]).area.sum() - last
        last += a
        mirrors[i - 1].area = a


def main():
    date = get_all_time()

    sun = [Sun(time) for time in date]
    tower = [Tower(s) for s in sun]
    mirror_list = [[Mirror(t, x[i], y[i]) for i in range(len(x))] for t in tower]

    data = process_map(compute, sun, tower, mirror_list, chunksize=1)
    # data = [compute(i) for i in tqdm.tqdm(date)]

    pd = pandas.DataFrame(
        columns=["日期", "平均光学效率", "平均余弦效率", "平均遮挡效率", "平均截断效率", "单位面积镜面年平均输出热功率"]
    )

    for i in data:
        pd.loc[len(pd)] = i

    pd.to_excel("result.xlsx", index=False)


def get_all_time():
    date = []
    for month in range(1, 13):
        for hour, minute in [(9, 0), (10, 30), (12, 0), (13, 30), (15, 0)]:
            date.append(datetime.datetime(2023, month, 21, hour, minute))
    return date


def draw():
    date = []
    for month in range(1, 13):
        for hour, minute in [(9, 0), (10, 30), (12, 0), (13, 30), (15, 0)]:
            date.append(datetime.datetime(2023, month, 21, hour, minute))

    dat = []

    # for time in date:
    #     sun = Sun(time)
    #     dat.append([sun.alpha, sun.gamma])
    #
    # dat = numpy.array(dat).T
    # i = 1
    # for line in dat[1].reshape(12, 5):
    #     plt.plot(
    #         [
    #             f"{hour}时{minute}分"
    #             for hour, minute in [(9, 0), (10, 30), (12, 0), (13, 30), (15, 0)]
    #         ],
    #         numpy.degrees(line),
    #         label=f"{i}月",
    #     )
    #     i += 1
    # plt.xlabel("天数")
    # plt.ylabel("太阳方位角")
    # plt.legend()
    # plt.savefig("img/太阳方位角.png")
    # plt.show()
    #
    # pass
    time = datetime.datetime(2023, 1, 21, 9, 0)
    sun = Sun(time)
    tower = Tower(sun)
    mirror_list = [Mirror(tower, x[i], y[i]) for i in range(len(x))]

    # mirror_list = mirror_list[100]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(0, 0, 0, ".", color="r")

    # ax.plot(*tower.coordinate, "o", color="b")

    # 圆柱
    # theta = numpy.linspace(0, 2 * numpy.pi, 1000)
    # z_ = numpy.linspace(0, 20, 1000)
    # z_ = numpy.linspace(0, 20, 1000)
    # theta, z_ = numpy.meshgrid(theta, z_)
    # x_ = tower.r * numpy.cos(theta)
    # y_ = tower.r * numpy.sin(theta)
    # ax.plot_surface(x_, y_, z_, alpha=0.4)

    v = numpy.array([*tower.project, tower.project[0]])
    ax.plot(v.T[0], v.T[1], v.T[2], color="b")
    # v = numpy.array([*tower.vertex, tower.vertex[0]])
    # ax.plot(v.T[0], v.T[1], v.T[2], color="g")

    for mirror in mirror_list[::]:
        v = numpy.array([*mirror.project, mirror.project[0]])
        ax.plot(v.T[0], v.T[1], v.T[2], color="r", alpha=0.8)
        # ax.plot(
        #     *mirror.coordinate,
        #     ".",
        # )

    # z轴尺寸
    ax.set_zlim(0, 5)

    # 锁定视角
    ax.view_init(azim=90, elev=90)

    # 可视区域
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    # 关闭网格
    ax.grid(False)

    # 关闭坐标轴
    # ax.set_axis_off()

    # 减小空白
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.title(f"{time.strftime('%Y年%m月%d日 %H时%M分 定日镜布局')}")

    # plt.savefig(f"see.png")
    plt.show()
    # plt.close()
    pass


if __name__ == "__main__":
    data = pandas.read_excel("附件.xlsx")
    x = data["x坐标 (m)"]
    y = data["y坐标 (m)"]

    main()
    # draw()
    pass
