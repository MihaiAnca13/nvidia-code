import tensorflow as tf
import matplotlib.pyplot as plt
from statistics import mean

plt.style.use('ggplot')

tags = ['val_loss', 'val_mean_absolute_error', 'val_mean_squared_error', 'loss', 'mean_absolute_error']
tag = tags[3]


# alexnet
a = []

for e in tf.train.summary_iterator('Graph/second-train/events.out.tfevents.1550408077.MIHAI-DESKTOP'):
    for v in e.summary.value:
        if v.tag == tag:
            a.append(v.simple_value)

avg_step = 10
y1 = []
for i in range(0, len(a), avg_step):
    y1.append(mean(a[i:i+avg_step]))

# pilotnet
a = []


for e in tf.train.summary_iterator('Graph/pilot-first-train/events.out.tfevents.1550770631.MIHAI-DESKTOP'):
    for v in e.summary.value:
        if v.tag == tag:
            a.append(v.simple_value)

avg_step = 10
y2 = []
for i in range(0, len(a), avg_step):
    y2.append(mean(a[i:i+avg_step]))

# seresnet
a = []

for e in tf.train.summary_iterator('Graph/seresnet-first-train/events.out.tfevents.1550919477.MIHAI-DESKTOP'):
    for v in e.summary.value:
        if v.tag == tag:
            a.append(v.simple_value)

avg_step = 10
y3 = []
for i in range(0, len(a), avg_step):
    y3.append(mean(a[i:i+avg_step]))

x = range(1, len(a), avg_step)

# plt.plot(x, y1)
plt.plot(x, y2)
# plt.plot(x, y3)
# plt.scatter(x, y1)
# plt.scatter(x, y2, marker='d')
# plt.scatter(x, y3, marker='*')
plt.title('Training')
plt.xlabel('Epochs')
plt.ylabel(tag)
# plt.legend(['AlexNET', 'PilotNET', 'SENet'])
plt.show()
