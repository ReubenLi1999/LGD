# LGD和地表质量变化的关系

将地表划分为*N*块区域，第*i*块区域地表质量变化引起的第*j*颗卫星重力异常表示为：
$$
\delta \boldsymbol{a}_j(t)=\sum_{i=1}^N \delta \boldsymbol{a}_{j,i}(t)=G\sum_{i=1}^N \boldsymbol{f}_{j,i}(t)\delta m_i(t)
$$
其中，$\boldsymbol{f}_{j,i}(t)=\frac{\boldsymbol{q}_i(t)-\boldsymbol{r}_j(t)}{ \vert \boldsymbol{q}_i (t)-\boldsymbol{r}_j(t) \vert ^3}$.

两颗卫星的重力异常矢量差为：
$$
\delta \boldsymbol{a}_{12}(t)=G\sum_{i=1}^N(\boldsymbol{f}_{1,i} - \boldsymbol{f}_{2,i})\delta m_i(t)=G(\boldsymbol{F}_1(t) - \boldsymbol{F}_2(t)) \delta \boldsymbol{m}(t)
$$
LGD是双星重力矢量差在星间连线方向的投影：
$$
\delta \boldsymbol{a}^{LOS}_{12}=\boldsymbol{e}_{12}(t)\cdot\delta \boldsymbol{a}_{12}(t)=\boldsymbol{e}_{12}(t)\cdot G\sum_{i=1}^N(\boldsymbol{f}_{1,i} - \boldsymbol{f}_{2,i})\delta m_i(t)=\boldsymbol{e}_{12}(t)\cdot G(\boldsymbol{F}_1(t) - \boldsymbol{F}_2(t)) \delta \boldsymbol{m}(t)
$$
