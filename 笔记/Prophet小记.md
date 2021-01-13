# 时间序列模型-Prophet

## 1. 介绍

Prophet是Facebook在2017年开源的一个时间序列预测算法，跟ARIMA模型不同的地方在于，从总体来看Prophet算法相当于一个时间序列模型和机器学期模型的一个组合，能更好的去应对噪声的干扰因素。并且和ARIMA模型输出为一个确定值不同，prophet最终输出的是一个置信区间，也就是合理的上界和下界。其可以预测的时间任务有以下特征：

- 对于历史在至少几个月（最好是一年）的每小时、每天或每周的观察
- 强大的多次的「人类规模级」的季节性：每周的一些天和每年的一些时候
- 事先知道的以不定期的间隔发生的重要节假日（如，双十一）
- 合理数量的缺失的观察或大量异常
- 历史趋势的改变，比如因为产品发布或记录变化
- 非线性增长曲线的趋势，其中有的趋势达到了自然极限或饱和。

![prophet](https://gitee.com/zhanghang23/picture_bed/raw/master/time%20series%20database/prophet.png)

上图为一个prophet的结果，黑点表示原始时间序列离散点，深蓝色的线表示使用时间序列来拟合所得的取值，浅蓝色的先表示时间序列的一个置信区间。

## 2. 原理

Prophet把时间序列分解成季节项 $S_t$，趋势项 $T_t$，剩余项 $R_t$，则对于所有的$t\geq 0 $，都有
$$
y_t = S_t + T_t +R_t
$$
或者
$$
y_t = S_t \times  T_t \times R_t
$$
以上两个表达式等价于$\ln y_t=\ln S_t + \ln T_t + \ln R_t$。所以有时候的预测模型要先取对数在进行时间序列的分解，就能得到乘法的形式。而除了以上的项之外，还添加的第四项是节假日效应，所以Prophet的表达形式为：
$$
y(t)=g(t)+s(t)+h(t)+\epsilon _t
$$
其中$g(t)$表示时间序列在非周期上面的趋势项；$s(t)$表示以年或者周，月为单位的周期，或者季节项；$h(t)$表示节假日项，表示当前是否存在节假日；$\epsilon _t$表示误差项或者剩余项。Prophet通过拟合这些项，最后累加起来得到最终的与时间序列预测值。

### 2.1 趋势项 $g(t)$

$g(t)$是趋势函数，表示时间序列上的非周期变化，在prophet中有两种形式：
$$
g(t)=\frac{C(t)}{1+\exp (-(k+a(t)^{\top }\delta )(t-(m+a(t)^{\top }\gamma )))}
$$

$$
g(t)=-(k+a(t)^{\top }\delta )t-(m+a(t)^{\top }\gamma )
$$

第一个式子用于拟合非线性变化的增长趋势，$C(t)$ 表示增长的上限，是 $t$ 的函数，认为在不同时间段序列增长的上限可能是不同的。其中$-(k+a(t)^{\top }\delta $ 表示增长速率，$m+a(t)^{\top }\gamma$ 表示线性的偏移。$a(t)$ 是一个指示函数，取值是 $\left \{  0,1\right \}$。引入$a(t)$ 的目的是考虑到时间序列中可能的突变点，这些突变点会对趋势函数有所影响，$\delta$ 和 $\gamma$ 表示突变点对趋势函数的斜率和偏移量影响的大小。算法中默认$\delta \sim Laplace(0,\tau )$, 由参数$\tau$ 控制模型改变斜率的灵活度。Porphet算法可以自动检测这些突变点，也可以手动传入这些突变点。Prophet也给了对应的接口来接受突变点：

```python
m = Prophet(changepoints=['2014-01-01'])
```

### 2.2 周期项 $s(t)$

$s(t)$用于刻画时间序列中的周期性（季节性）变化，其形式为傅里叶级数：
$$
s(t)=\sum_{N}^{n=1}(a_{n}\cos (\frac{2\pi nt}{P})+b_{n}\sin (\frac{2\pi nt}{P}))
$$
$P$ 是时间周期，当$P=7$时刻画的是以周为周期，$P=365.25$ 则是年。$a_n$和$b_n$是需要学习的参数。由傅里叶级数的性质可知，$N$ 越大越能刻画变化多的周期性模式，默认使用$N=10$ 以年为单位的周期性变化，$N=3$ 刻画以周为单位的周期性变化。

### 2.3 节假日项 $h(t)$

$h(t)$ 用于拟合节假日和特殊日期，比如中国的双十一，美国的黑五，超级碗等等，函数形式为：
$$
h(t)=Z(t)\kappa
$$
用以表示holiday的时长，$\kappa$ 参数用来刻画在整个趋势的改变程度。 $\kappa \sim Normal(0,v^2)$ 并且该正态分布是受到$v$ =holidays_prior_scale 这个指标影响的。默认是10，当值越大表示节假日对模型影响越大；当越小表示节假日对模型影响效果越小。

### 2.4 模型拟合

Prophet采用了L-BGFS的方式进行优化秋的各个参数的最大后验估计。

## 3. 使用Prophet和调参

在 Prophet 中，用户一般可以设置以下四种参数：

1. Capacity：在增量函数是逻辑回归函数的时候，需要设置的容量值。
2. Change Points：可以通过 n_changepoints 和 changepoint_range 来进行等距的变点设置，也可以通过人工设置的方式来指定时间序列的变点。
3. 季节性和节假日：可以根据实际的业务需求来指定相应的节假日。
4. 光滑参数： $\tau$ =changepoint_prior_scale 可以用来控制趋势的灵活度，  $\sigma$= seasonality_prior_scale 用来控制季节项的灵活度， $v$= holidays prior scale 用来控制节假日的灵活度。

如果不想设置的话，使用 Prophet 默认的参数即可

### 3.1 建立模型

首先导入Prophet的包

```python
from fbprophet import Prophet
```

利用pandas导入数据，然后由于关键字原因需要将时间戳和数据进行改名为ds和y

```python
df = df.rename(columns={'timestamp':'ds', 'value':'y'})
```

然后进行初始化模型，拟合模型，然后就可以进行时间序列预测了。

```python
初始化模型：m = Prophet()
拟合模型：m.fit(df)
计算预测值：periods 表示需要预测的点数，freq 表示时间序列的频率。
future = m.make_future_dataframe(periods=30, freq='min')
future.tail()
forecast = m.predict(future)
```

Prophet中有两个增长函数，分别是分段线性函数和逻辑回归函数。而 m = Prophet() 默认使用的是分段线性函数（linear），并且如果要是用逻辑回归函数的时候，需要设置 capacity 的值，i.e. df['cap'] = 100，否则会出错。

```python
m = Prophet()
m = Prophet(growth='linear')
m = Prophet(growth='logistic')
```



### 3.2 预测结果可视化

在建立模型之后，Prophet自带的可视化就可以很方便的将预测结果plot出来。

```python
画出预测图：
m.plot(forecast)
画出时间序列的分量：
m.plot_components(forecast)
```



![prophet_plot](https://gitee.com/zhanghang23/picture_bed/raw/master/time%20series%20database/prophet.png)

还可以对预测的结果进行成分分析展示其中的趋势，周筱盈，年效应

```python
model.plot_components(forecast)
```

![components](https://gitee.com/zhanghang23/picture_bed/raw/master/time%20series%20database/trend.png)

有些场景的数值会有一个限制，我们可以通过添加饱和增长的形式来限制其趋势变化的上界或下届。

```python
# 饱和增长
df['cap'] = 8.5
m = Prophet(growth='logistic')
m.fit(df)

# 预测未来 3 年的数据
future = m.make_future_dataframe(periods=1826)
# 将未来的承载能力设定得和历史数据一样
future['cap'] = 8.5
fcst = m.predict(future)
fig = m.plot(fcst)
```

![capacity](https://gitee.com/zhanghang23/picture_bed/raw/master/time%20series%20database/cap.png)
