from trafficdl.model.abstract_model import AbstractModel
import torch
from torch import nn
from torch.nn import functional as f
from sklearn.cluster import DBSCAN
from datetime import datetime, date, time
from haversine import haversine
import math
from copy import copy
from glob import glob

'''
18231047王肇凯 18231051朱乐岩
'''


class Point(object):

    def __init__(self, lat, long=None, the_date=None, the_time=None):
        if isinstance(lat, Point):
            self.lat = lat.lat
            self.long = lat.long
            self.date = lat.date
            self.time = lat.time
            self.datetime = lat.datetime
        elif isinstance(the_date, datetime):
            self.lat = float(lat)
            self.long = float(long)
            self.datetime = the_date
        else:
            self.lat = float(lat)
            self.long = float(long)
            d = list(map(int, the_date.split('-')))
            t = list(map(int, the_time.split(':')))
            dt = d + t
            self.time = time(*t)
            self.date = date(*d)
            self.datetime = datetime(*dt)

    def __sub__(self, p):
        if not isinstance(p, Point):
            raise Exception("Unexpected type of argument, expected Point, got %s." % type(p))
        return haversine((self.lat, self.long), (p.lat, p.long)), math.fabs(
            (self.datetime - p.datetime).total_seconds())

    def __repr__(self, **kwargs):
        return '({},{})@{}'.format(self.lat, self.long, str(self.datetime))

    def __str__(self, **kwargs):
        return self.__repr__(**kwargs)

    def __eq__(self, p):
        if not isinstance(p, Point):
            return False
        return self.lat == p.lat and self.long == p.long

    def __hash__(self, **kwargs):
        return hash(self.lat) + hash(self.long)


class StayPoint(object):
    def __init__(self, lat, long, arv_time, lev_time):
        self.lat = lat
        self.long = long
        self.arv_time = arv_time
        self.lev_time = lev_time

    def __repr__(self):
        return "({}, {}, {})".format(self.lat, self.long,
                                     int(math.fabs((self.arv_time -
                                                    self.lev_time)
                                                   .total_seconds())))


class Trajectory(object):

    def __init__(self, points=None, **kwargs):
        if points is None:
            self.points = []
        elif isinstance(points, list):
            assert all([isinstance(p, Point) for p in points])
            self.points = copy(points)
        else:
            raise Exception("Unsupported type %s." % type(points))

    def add(self, p):
        if not isinstance(p, Point):
            raise Exception("Unsupported type, expected Point, got %s." % type(p))
        self.points.append(p)

    def insert(self, index, p):
        if not isinstance(p, Point):
            raise Exception("Unsupported type, expected Point, got %s." % type(p))
        self.points.insert(index, p)

    def remove_at(self, index):
        del self.points[index]

    def remove(self, p):
        if not isinstance(p, Point):
            raise Exception("Unsupported type, expected Point, got %s." % type(p))
        self.points.remove(p)

    def clear(self):
        self.points = []

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, index):
        return self.points[index]

    def __len__(self):
        return len(self.points)

    def __repr__(self, **kwargs):
        p = [str(p_) for p_ in self.points]

        return "[" + "=>".join(p) + "]"

    def __str__(self, **kwargs):
        return self.__repr__(**kwargs)


def read_data(data_path):
    traj = Trajectory()
    for file_name in glob(data_path):
        with open(file_name) as fin:

            lines = list(fin)

            for line in lines[6:]:
                vals = line.split(',')
                traj.add(Point(vals[0], vals[1], vals[-2], vals[-1]))
    return traj


def get_interest_point_candidates(trajectory, min_dist, min_time):
    i = 0
    point_num = len(trajectory)
    interest_points = []
    while i < point_num:
        j = i + 1
        token = False
        pi = trajectory[i]
        while j < point_num:
            pj = trajectory[j]
            dist, tm = pi - pj
            if dist > min_dist:
                if tm > min_time:
                    l1 = 0.0
                    g = 0.0
                    n = 0.0
                    for p in trajectory[i:j + 1]:
                        l1 += p.lat
                        g += p.long
                        n += 1
                    l1 /= n
                    g /= n
                    interest_points.append(StayPoint(l1, g, pi.datetime, pj.datetime))
                    i = j
                    token = True
                break
            j += 1
        if not token:
            i += 1
    return interest_points


def prpare_data(traj, window_size=3, min_dist=0.2, min_time=1800):
    candidate_points = get_interest_point_candidates(traj, min_dist, min_time)
    s = [(p.lat, p.long) for p in candidate_points]
    db = DBSCAN(eps=0.1, min_samples=10).fit(s)
    point2lbl = {}
    for l2, p in zip(db.labels_, candidate_points):
        l2 += 1
        point2lbl[p] = l2
    time_input = []
    space_input = []
    p = candidate_points[0]
    curr_label = point2lbl[p]
    for p in candidate_points:
        l2 = point2lbl[p]
        if l2 != curr_label:
            space_input.append(l2)
            time_input.append(p.lev_time.hour - 1)
            curr_label = l2
    n = len(space_input)
    space_inputs = []
    time_inputs = []
    outputs = []
    for i in range(n - window_size):
        space_inputs.append(space_input[i: i + window_size])
        time_inputs.append(time_input[i: i + window_size])
        outputs.append(space_input[i + window_size])
    return space_inputs, time_inputs, outputs, len(set(db.labels_))


class STFRNN(AbstractModel):
    def __init__(self, config, data_feature):
        super(STFRNN, self).__init__(config, data_feature)
        self.config = config
        self.emb1 = nn.Embedding(self.config["nb_points"], self.config["embedding_size_1"])
        self.emb2 = nn.Embedding(self.config["tm_length"], self.config["embedding_size_2"])
        self.rnn = nn.RNN(
            input_size=self.config["embedding_size_1"] + self.config["embedding_size_2"],
            hidden_size=self.config["rnn_size"]
        )
        self.dense = nn.Linear(self.config["rnn_size"], self.config["dense_dim"])

    def forward(self, batch):
        s_input = batch["history_loc"]
        t_input = batch["history_tim"]
        s_input = torch.zeros((batch["target"].shape[0], self.config["dense_dim"]), dtype=torch.long).to(
            self.config["device"])
        t_input = torch.zeros((batch["target"].shape[0], self.config["dense_dim"]), dtype=torch.long).to(
            self.config["device"])
        xe = self.emb1(s_input)
        he = self.emb2(t_input)

        x = torch.cat((xe, he), dim=-1)
        x = self.rnn(x)
        y = f.softmax(self.dense(x[0]))
        return y

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        criterion = nn.CrossEntropyLoss()
        categories = self.forward(batch)

        target = (batch['target'] / (batch['target'].max() + 1) * self.config["nb_points"]) \
            .to(torch.long)

        return criterion(categories[..., 0], target)
