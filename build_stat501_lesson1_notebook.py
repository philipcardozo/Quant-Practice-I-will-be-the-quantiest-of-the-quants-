from pathlib import Path
import nbformat as nbf
nb=nbf.v4.new_notebook()
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}}
def md(x): nb.cells.append(nbf.v4.new_markdown_cell(x.strip()))
def c(x): nb.cells.append(nbf.v4.new_code_cell(x.strip()))

md(r"""
# STAT 501 Lesson 1: Simple Linear Regression in Python
## Professional study notebook

This executable notebook covers all substantive sections of [Penn State STAT 501 Lesson 1](https://online.stat.psu.edu/stat501/Lesson01): definitions, least squares, the population model and LINE conditions, error variance, R-squared, correlation, all seven cautions, correlation inference, and applied examples.

The included examples are reproducible offline. They use the course height/weight values, course-reported numerical anchors, and seeded illustrative reconstructions. If you added course data files to the folder, the loader cell will find them.

Source: Penn State STAT 501 Lesson 1, accessed October 2026. This is a Python instructional companion, not an official course handout.
""")
md(r"""
## Learning map

| Course topic | Notebook application |
|---|---|
| 1.1 | Predictor/response; deterministic versus statistical relationships |
| 1.2 | Residuals, OLS criterion, formulas, slope/intercept, extrapolation |
| 1.3-1.4 | Population line, LINE assumptions, MSE and residual standard error |
| 1.5-1.7 | SSTO = SSR + SSE, R-squared, Pearson r, interpretation examples |
| 1.8 | Seven cautions, each demonstrated with code/plots |
| 1.9 | t test for population correlation rho |
| 1.10 | Poverty/birth-rate and FEV/age style analyses |
| End | Five interview questions with progressively harder solutions |
""")
c(r"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import statsmodels.api as sm
from scipy import stats
RNG=np.random.default_rng(501)
sns.set_theme(style="whitegrid",context="notebook")
plt.rcParams.update({"figure.figsize":(8,4.8),"axes.titleweight":"bold"})

def fit(x,y):
    return sm.OLS(np.asarray(y,float),sm.add_constant(np.asarray(x,float))).fit()
def metric(x,y,m=None):
    x,y=np.asarray(x,float),np.asarray(y,float); m=fit(x,y) if m is None else m
    yh=np.asarray(m.fittedvalues); sse=((y-yh)**2).sum(); ssr=((yh-y.mean())**2).sum(); ssto=((y-y.mean())**2).sum()
    return pd.Series({"n":len(y),"b0":m.params[0],"b1":m.params[1],"SSE":sse,"SSR":ssr,"SSTO":ssto,
      "MSE=SSE/(n-2)":sse/(len(y)-2),"residual SE s":np.sqrt(sse/(len(y)-2)),"R²":ssr/ssto,"r":np.corrcoef(x,y)[0,1]})
def plot_line(x,y,m,xlab="x",ylab="y",title=""):
    o=np.argsort(x); plt.scatter(x,y,alpha=.8,label="observations")
    plt.plot(np.asarray(x)[o],np.asarray(m.fittedvalues)[o],color="crimson",lw=2.5,label="OLS line")
    plt.xlabel(xlab);plt.ylabel(ylab);plt.title(title);plt.legend();plt.show()

names=["student_height_weight.txt","skincancer.txt","poverty.txt","fev_dat.txt","heightgpa.txt","signdist.txt","bldgstories.csv"]
found=[p.name for p in Path(".").rglob("*") if p.name in names]
print("Course data found:",found or "none; self-contained examples will run.")
""")
md(r"""
---
# 1.1 What is simple linear regression?

Simple linear regression describes two continuous quantitative variables. X is the predictor (also explanatory or independent variable); Y is the response (also outcome or dependent variable). It is simple because it has one predictor.

A deterministic relationship is exact. A statistical relationship has a trend plus scatter. Regression is for the latter.
""")
c(r"""
C=np.arange(0,51,5); F=32+1.8*C
height=np.array([63,64,66,69,69,71,71,72,73,75])
weight=np.array([127,121,142,157,162,156,169,165,181,208])
fig,ax=plt.subplots(1,2,figsize=(13,4.5))
ax[0].plot(C,F,"o-");ax[0].set(title="Deterministic: F = 32 + 1.8C",xlabel="Celsius",ylabel="Fahrenheit")
ax[1].scatter(height,weight,s=65,color="tab:orange");ax[1].set(title="Statistical: height and weight",xlabel="Height (in)",ylabel="Weight (lb)")
plt.tight_layout()
""")
md(r"""
**Why it matters:** inspect a labeled scatterplot before fitting. It shows direction, form, unusual points, and changing spread. The lesson's skin-cancer/latitude relationship, like height/weight, trends but is not exact.
""")
md(r"""
---
# 1.2 Best fitting line and least squares

For observation i, the fitted value and residual are

$$\hat y_i=b_0+b_1x_i,\qquad e_i=y_i-\hat y_i.$$

Ordinary least squares selects the line minimizing the sum of squared residuals:

$$Q(b_0,b_1)=\sum_{i=1}^n[y_i-(b_0+b_1x_i)]^2=\sum e_i^2.$$

Squares prevent positive and negative errors from cancelling and penalize larger misses more strongly.
""")
c(r"""
candidate_b0,candidate_b1=-331.2,7.1
cand=candidate_b0+candidate_b1*height
hw=fit(height,weight)
table=pd.DataFrame({"height":height,"weight":weight,"candidate fitted":cand,"OLS fitted":hw.fittedvalues,
                    "candidate residual²":(weight-cand)**2,"OLS residual²":(weight-hw.fittedvalues)**2})
display(table.round(2))
print("Candidate SSE:",round(table["candidate residual²"].sum(),2)," | OLS SSE:",round(table["OLS residual²"].sum(),2))
g=np.linspace(height.min(),height.max(),100)
plt.scatter(height,weight,s=65,color="black");plt.plot(g,candidate_b0+candidate_b1*g,"--",label="candidate")
plt.plot(g,hw.params[0]+hw.params[1]*g,color="crimson",lw=2.5,label="OLS")
for x,y,yh in zip(height,weight,hw.fittedvalues): plt.vlines(x,y,yh,color="gray",alpha=.6)
plt.title("OLS minimizes total squared vertical residuals");plt.xlabel("Height");plt.ylabel("Weight");plt.legend();plt.show()
""")
md(r"""
The formulas are

$$b_1={\sum(x_i-\bar x)(y_i-\bar y)\over\sum(x_i-\bar x)^2},\qquad b_0=\bar y-b_1\bar x.$$

The denominator is positive, so the cross-products determine the slope sign. The OLS line passes through $(\bar x,\bar y)$.
""")
c(r"""
xb,yb=height.mean(),weight.mean()
b1=((height-xb)*(weight-yb)).sum()/((height-xb)**2).sum();b0=yb-b1*xb
print(f"Manual: b0={b0:.4f}, b1={b1:.4f}; software: b0={hw.params[0]:.4f}, b1={hw.params[1]:.4f}")
print(f"Centroid ({xb:.2f},{yb:.2f}); fitted at x-bar={b0+b1*xb:.2f}")
display(pd.DataFrame({"height inches":[66,67],"predicted mean weight lb":(b0+b1*np.array([66,67])).round(2)}))
""")
md(r"""
### Interpretation

- **Slope:** In the observed range, a one-inch greater height is associated with about 6.14 lb greater **predicted mean** weight.
- **Intercept:** predicted mean Y at X = 0. Here a 0-inch person is outside scope, so it is not scientifically meaningful.
- **Extrapolation:** do not trust this line at heights far beyond 63-75 inches.
""")
md(r"""
---
# 1.3 Population model and LINE conditions

OLS estimates an unknown population relationship:

$$Y_i=\underbrace{\beta_0+\beta_1x_i}_{E(Y_i\mid x_i)}+\varepsilon_i.$$

The population line is $E(Y|x)=\beta_0+\beta_1x$; sample coefficients $b_0,b_1$ estimate $\beta_0,\beta_1$. A different sample produces a different fitted line.
""")
c(r"""
gpa_pop=np.repeat(np.arange(1,5),100);score_pop=2+4*gpa_pop+RNG.normal(0,1,400)
ix=np.concatenate([RNG.choice(np.where(gpa_pop==z)[0],3,replace=False) for z in range(1,5)])
gpa,score=gpa_pop[ix],score_pop[ix];gf=fit(gpa,score);g=np.linspace(1,4,100)
plt.scatter(gpa_pop,score_pop,alpha=.12,s=15,label="population individuals");plt.plot(g,2+4*g,color="black",lw=2.5,label="population mean")
plt.scatter(gpa,score,s=70,color="crimson",label="sample n=12");plt.plot(g,gf.params[0]+gf.params[1]*g,"--",color="crimson",lw=2.5,label="sample OLS")
plt.xlabel("GPA");plt.ylabel("College score");plt.title("Population line versus one sample estimate");plt.legend();plt.show()
""")
md(r"""
## LINE assumptions

| Letter | Condition | Meaning |
|---|---|---|
| L | Linear function | $E(Y|x)$ is linear; errors average zero at each x |
| I | Independent | one error does not reveal another |
| N | Normally distributed | errors at each x are normal, important for small-sample inference |
| E | Equal variances | $Var(\varepsilon|x)=\sigma^2$ is constant |

Equivalently, errors are independent normal random variables with mean zero and common variance $\sigma^2$. These are assumptions about conditional errors, not merely raw X or Y distributions.
""")
md(r"""
---
# 1.4 Common error variance, MSE, and residual standard error

$\sigma^2$ measures individual response variation around the unknown population mean line. Its estimate is

$$MSE={SSE\over n-2}={\sum(y_i-\hat y_i)^2\over n-2},\qquad s=\sqrt{MSE}.$$

MSE estimates $\sigma^2$, and residual standard error s has Y units. We divide by n-2 because regression estimates two parameters. Compare single-population sample variance, which divides by n-1 after estimating one mean.
""")
c(r"""
display(metric(height,weight,hw)[["SSE","MSE=SSE/(n-2)","residual SE s","R²"]].to_frame("height to weight").round(3))
tc=np.linspace(0,40,18);A=32+1.8*tc+RNG.normal(0,5,18);B=32+1.8*tc+RNG.normal(0,1,18)
fa,fb=fit(tc,A),fit(tc,B);fig,ax=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
for yy,mm,a,label in [(A,fa,ax[0],"A noisier"),(B,fb,ax[1],"B more precise")]:
 a.scatter(tc,yy);a.plot(tc,mm.fittedvalues,color="crimson");a.set(title=f"{label}; s={np.sqrt(mm.mse_resid):.2f} F",xlabel="Celsius",ylabel="Fahrenheit")
plt.tight_layout()
""")
md(r"""
---
# 1.5 R-squared and 1.6 Pearson correlation

OLS partitions variation:

$$SSTO=\sum(y_i-\bar y)^2=SSR+SSE,\qquad R^2={SSR\over SSTO}=1-{SSE\over SSTO}.$$

For an intercept model, $0\le R^2\le1$. Say: **accounting for X reduces sample variation in Y by 100R-squared percent.** This is not a causal statement.

In simple regression:

$$r=\operatorname{sign}(b_1)\sqrt{R^2}={\sum(x_i-\bar x)(y_i-\bar y)\over\sqrt{\sum(x_i-\bar x)^2\sum(y_i-\bar y)^2}}.$$

r is unitless, symmetric, and between -1 and 1. Its sign matches the slope. It measures linear direction/strength only: r=0 means no linear association, not necessarily no relationship.
""")
c(r"""
z=metric(height,weight,hw)
print(f"SSTO={z['SSTO']:.3f}; SSR+SSE={z['SSR']:.3f}+{z['SSE']:.3f}={z['SSR']+z['SSE']:.3f}")
print(f"slope={hw.params[1]:.3f}; r={z['r']:.3f}; r²={z['r']**2:.3f}; R²={z['R²']:.3f}")
print("Correlation unchanged after inches -> cm:",round(np.corrcoef(height*2.54,weight)[0,1],6))
""")
md(r"""
## 1.7 Lesson interpretation anchors

| Example | Course values | Meaning |
|---|---|---|
| Celsius/Fahrenheit | r=1, R-squared=100% | perfect deterministic positive line |
| Building stories/height | r about .951, R-squared about 90.4% | very strong positive linear association |
| Driver age/seeing distance | r about -.801, R-squared about 64.2% | fairly strong negative linear association |
| Student height/GPA | r about -.053, R-squared about 0.3% | essentially no linear association |

No universal cutoff makes R-squared or r “good.” Domain, decision stakes, and error costs matter.
""")
md(r"""
---
# 1.8 Seven cautions about R-squared and r
""")
md(r"""
### 1. Linear association only
A perfect curve can have r=0 and linear R-squared=0. Always plot.
""")
c(r"""
x=np.linspace(-3,3,81);y=x**2;lin=fit(x,y);quad=sm.OLS(y,np.column_stack([np.ones_like(x),x,x**2])).fit()
plt.scatter(x,y,s=18);plt.plot(x,lin.fittedvalues,color="crimson",label="linear");plt.plot(x,quad.fittedvalues,color="green",label="quadratic")
plt.title(f"Perfect curve: r={np.corrcoef(x,y)[0,1]:.3f}, linear R²={lin.rsquared:.3f}, quadratic R²={quad.rsquared:.3f}");plt.legend();plt.show()
""")
md(r"""
### 2. Large R-squared does not prove a line fits well
It only says a line is better than a mean-only model. It can still miss curvature.
""")
c(r"""
year=np.arange(1790,2000,10);pop=3.9*np.exp(.014*(year-1790))+RNG.normal(0,2,len(year));pf=fit(year,pop)
plt.scatter(year,pop);plt.plot(year,pf.fittedvalues,color="crimson",label=f"linear R²={pf.rsquared:.3f}")
plt.plot(year,np.polyval(np.polyfit(year,pop,2),year),color="green",label="quadratic trend");plt.title("High linear R² can conceal curvature");plt.legend();plt.show()
""")
md(r"""
### 3. One point can control the result
Check an influential observation; do not automatically delete it. Verify it, decide whether it is in target scope, and report sensitivity.
""")
c(r"""
xc=np.array([5,5.5,6,6.5,7]);yc=np.array([72,63,59,51,43]);xa=np.r_[xc,8.3];ya=np.r_[yc,503];ma,mc=fit(xa,ya),fit(xc,yc)
fig,ax=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
for a,xx,yy,mm,label in [(ax[0],xa,ya,ma,"with extreme point"),(ax[1],xc,yc,mc,"without it")]:
 o=np.argsort(xx);a.scatter(xx,yy,s=65);a.plot(xx[o],mm.fittedvalues[o],color="crimson");a.set(title=f"{label}: slope={mm.params[1]:.1f}, r={np.corrcoef(xx,yy)[0,1]:.3f}",xlabel="magnitude",ylabel="deaths")
plt.tight_layout()
""")
md(r"""
### 4. Association is not causation
Reverse causation, confounding, selection, and time trends can create association. The course's drug-budget/deaths exercise makes this point. Randomized experiments offer much stronger causal evidence.
""")
c(r"""
yrs=np.arange(1990,2010);budget=20+2.2*(yrs-1990)+RNG.normal(0,3,20);deaths=50+1.4*(yrs-1990)+RNG.normal(0,7,20)
fig,ax=plt.subplots(1,2,figsize=(13,4.5));ax[0].scatter(budget,deaths);ax[0].set(title=f"Budget/deaths r={np.corrcoef(budget,deaths)[0,1]:.2f}",xlabel="budget",ylabel="deaths")
ax[1].plot(yrs,budget,"o-",label="budget");ax[1].plot(yrs,deaths,"o-",label="deaths");ax[1].set(title="Year can drive both series");ax[1].legend();plt.tight_layout()
""")
md(r"""
### 5. Ecological correlations can overstate association
Group means answer a group-level question. Do not use them to make individual claims (ecological fallacy).
""")
c(r"""
reg=np.repeat(np.arange(9),70);re=np.repeat(np.linspace(10,16,9),70);edu=re+RNG.normal(0,1.8,len(reg));income=15+2*edu+5*re+RNG.normal(0,18,len(reg))
means=pd.DataFrame({"reg":reg,"edu":edu,"income":income}).groupby("reg").mean()
fig,ax=plt.subplots(1,2,figsize=(13,4.5));ax[0].scatter(edu,income,alpha=.22);ax[0].set(title=f"Individuals r={np.corrcoef(edu,income)[0,1]:.2f}")
ax[1].scatter(means.edu,means.income,s=80);ax[1].set(title=f"Region means r={np.corrcoef(means.edu,means.income)[0,1]:.2f}");plt.tight_layout()
""")
md(r"""
### 6. Statistical significance is not practical significance
Huge samples can make tiny slopes have tiny p-values. Report units, interval, R-squared, and a decision threshold.
""")
c(r"""
xbig=RNG.normal(0,1,10000);ybig=5+.10*xbig+RNG.normal(0,1,10000);big=fit(xbig,ybig);ci=np.asarray(big.conf_int())[1]
print(f"slope={big.params[1]:.3f}; CI=({ci[0]:.3f},{ci[1]:.3f}); p={np.asarray(big.pvalues)[1]:.2g}; R²={big.rsquared:.4f}")
print("Tiny p does not make a 0.10-unit effect decision-useful.")
""")
md(r"""
### 7. Large R-squared does not guarantee useful predictions
Use a prediction interval for a new individual; it can be too wide to support a decision even with high R-squared.
""")
c(r"""
duration=RNG.uniform(1.6,5.1,21);wait=23+14*duration+RNG.normal(0,9,21);faith=fit(duration,wait)
display(faith.get_prediction(exog=[[1,3]]).summary_frame()[["mean","mean_ci_lower","mean_ci_upper","obs_ci_lower","obs_ci_upper"]].round(2))
print(f"R²={faith.rsquared:.3f}; obs_ci is for a future individual, mean_ci for an average.")
""")
md(r"""
---
# 1.9 Test for population correlation rho

When neither variable is naturally the response, test $H_0:\rho=0$ with

$$t^*={r\sqrt{n-2}\over\sqrt{1-r^2}},\qquad df=n-2.$$

For a two-sided test, compare to a t distribution and calculate $p=2P(T\ge|t^*|)$. If p is below alpha, conclude sufficient evidence of a linear population association. Do not say prove causation or accept the null.
""")
c(r"""
def corr_test(r,n,alt="two-sided"):
 df=n-2;t=r*np.sqrt(df)/np.sqrt(1-r*r)
 p=2*stats.t.sf(abs(t),df) if alt=="two-sided" else (stats.t.sf(t,df) if alt=="greater" else stats.t.cdf(t,df))
 return pd.Series({"r":r,"n":n,"df":df,"t*":t,"p-value":p})
display(corr_test(.939,170).to_frame("lesson husband/wife example").round(6))
""")
md(r"""
Conditions: random independent pairs from a bivariate-normal population; either orientation can be linear and conditional variance is constant. If predictor/response roles are clear, use the regression slope test (Lesson 2) instead.
""")
md(r"""
---
# 1.10 Further applications

## Poverty rate and teen birth rate
The course reports approximately $\hat y=4.267+1.373x$, s=5.55, R-squared=53.3% for state/DC aggregate data. A one percentage-point higher poverty rate is associated with 1.373 higher births per 1,000 females aged 15-17 on average. It is observational aggregate evidence, not individual-level causal evidence. The zero-poverty intercept is outside scope.
""")
c(r"""
poverty=RNG.uniform(7,25,51);births=4.267+1.373*poverty+RNG.normal(0,5.55,51);pfit=fit(poverty,births)
plot_line(poverty,births,pfit,"Poverty rate %","Births per 1,000","Illustrative poverty/birth-rate analysis")
display(metric(poverty,births,pfit)[["b0","b1","residual SE s","R²"]].to_frame("result").round(3))
""")
md(r"""
## Lung function FEV and age
For ages 6-10, the course reports approximately FEV = 0.0498 + 0.2621 age; at age 8, fitted mean FEV is 2.1466. The key feature is non-constant variance: FEV varies more at older ages. The full 3-19 data gives a noticeably different line, showing that range/subset choice matters.
""")
c(r"""
age=RNG.uniform(3,19,500);fev=.43+.222*age+RNG.normal(0,.12+.025*age,500);sub=(age>=6)&(age<=10);fs,ff=fit(age[sub],fev[sub]),fit(age,fev)
fig,ax=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
for a,xx,yy,m,label in [(ax[0],age[sub],fev[sub],fs,"ages 6-10"),(ax[1],age,fev,ff,"full ages 3-19")]:
 o=np.argsort(xx);a.scatter(xx,yy,s=13,alpha=.35);a.plot(xx[o],m.fittedvalues[o],color="crimson",lw=2);a.set(title=label,xlabel="Age",ylabel="FEV")
plt.suptitle("Range choice changes the line; spread widens with age",y=1.02);plt.tight_layout()
print("Subset fitted mean FEV at age 8:",round(fs.params[0]+8*fs.params[1],3))
""")
md(r"""
---
# Python workflow reference

| Goal | Python |
|---|---|
| Fit | sm.OLS(y, sm.add_constant(x)).fit() |
| Summary | fit.summary() |
| Fitted values and residuals | fit.fittedvalues and fit.resid |
| SSE, MSE, residual SE | sum(fit.resid squared), fit.mse_resid, sqrt(fit.mse_resid) |
| R-squared | fit.rsquared |
| Pearson correlation | stats.pearsonr(x,y) |
| Mean CI and individual PI | fit.get_prediction(...).summary_frame() |

**Workflow:** define question and units, plot, fit, interpret in range, report s/R-squared/r, examine LINE and unusual points, distinguish association from causation, then state decision relevance and uncertainty.
""")
md(r"""
---
# Interview practice: five questions with solutions

A strong answer includes calculation, interpretation in units, and a limitation or next check.
""")
md(r"""
## 1. Interpret a result
**Question:** Sales and ad spend are measured in thousands: $\hat{sales}=12+3.5(ad\ spend)$, R-squared=.64. Interpret coefficients and R-squared. Predict at spend=4. Is it causal?
""")
c(r"""
print("Prediction = 12 + 3.5(4) = 26 thousand dollars.")
print("Slope: +$3.5k predicted mean sales per additional $1k spend, within range.")
print("Intercept: predicted $12k at zero spend; assess if zero is meaningful.")
print("R-squared: spend reduces sample variation in sales by 64%.")
print("Not necessarily causal: spending may be targeted toward high-demand weeks.")
""")
md(r"""
## 2. Explain OLS
**Question:** Why square residuals? Derive a line by hand. What is a drawback?
""")
c(r"""
x2=np.array([1,2,3,4,5.]);y2=np.array([3,5,5,8,9.]);bb1=((x2-x2.mean())*(y2-y2.mean())).sum()/((x2-x2.mean())**2).sum();bb0=y2.mean()-bb1*x2.mean();e=y2-(bb0+bb1*x2)
print(f"b0={bb0:.3f}, b1={bb1:.3f}, SSE={(e**2).sum():.3f}")
print("Squares prevent cancellation and optimize smoothly; drawback: they amplify outlier influence.")
""")
md(r"""
## 3. Challenge a misleading summary
**Question:** An analyst says r=0 means machine temperature has no relationship to product quality, but the scatterplot is U-shaped. Respond.
""")
c(r"""
print("r=0 means no linear association; the U-shape can be a strong nonlinear relationship.")
print(f"Demonstration: linear R²={lin.rsquared:.3f}; quadratic R²={quad.rsquared:.3f}.")
print("Next: inspect residuals, fit a justified nonlinear term, and validate out of sample.")
""")
md(r"""
## 4. Test correlation
**Question:** In 30 random pairs r=-.45. At alpha=.05, test H0: rho=0 vs HA: rho not equal to 0.
""")
c(r"""
ans=corr_test(-.45,30);display(ans.to_frame("calculation").round(5))
print("p is below .05: reject H0. There is sufficient evidence of a nonzero linear population association.")
print("This is not a causal conclusion and relies on independence/random-sampling conditions.")
""")
md(r"""
## 5. Senior analyst recommendation
**Question:** 100,000 observational records show 1 engagement point associates with +0.03 percentage-point retention, p<.001, R-squared=.002. Should the company invest?
""")
c(r"""
print("\n".join([
"1. Separate statistical from practical significance: huge n makes tiny effects significant.",
"2. Convert a realistic intervention into retained users and dollars; compare to cost and a pre-set threshold.",
"3. Do not claim causality: reverse causation and confounding are plausible.",
"4. Check segments, time ordering, nonlinearity, uncertainty, and out-of-sample performance.",
"5. If causal ROI matters, run an A/B test or defensible quasi-experiment.",
"6. R-squared=.002 means engagement alone has limited predictive value."
]))
""")
md(r"""
---
# Takeaway sheet

1. A fitted line estimates a mean response, not individual certainty.
2. OLS minimizes SSE and passes through the data centroid.
3. MSE estimates common error variance; residual SE restores response units.
4. r and R-squared summarize sample linear association, not causation.
5. Scatterplots are non-negotiable: they expose curvature, changing spread, and influence.
6. Pair statistical results with scope, uncertainty, practical magnitude, and design limits.
""")
md(r"""
---
# Appendix: Analyze the supplied Penn State data files

The notebook deliberately stays runnable without downloads, but your workspace includes the original lesson height/weight and skin-cancer files. This optional cell uses them directly. It is a useful model for replacing any illustrative array above with a real course file.
""")
c(r"""
data_dir=Path("Regression")
skin_path=data_dir / "skincancer.txt"
height_path=data_dir / "student_height_weight.txt"
if skin_path.exists() and height_path.exists():
    skin=pd.read_csv(skin_path,sep=r"\s+",engine="python")
    hw_file=pd.read_csv(height_path,sep=r"\s+",engine="python")
    skin_fit=fit(skin["Lat"],skin["Mort"])
    print("Loaded supplied files:",skin_path.name,"and",height_path.name)
    plot_line(skin["Lat"],skin["Mort"],skin_fit,"Latitude at state center","Skin-cancer mortality per 10 million","Original Penn State skin-cancer data")
    display(metric(skin["Lat"],skin["Mort"],skin_fit)[["b0","b1","residual SE s","R²","r"]].to_frame("Lat -> Mort").round(3))
    print("Interpretation: higher latitude is associated with lower mortality in these state-level observational data.")
    print("This is an ecological association: it does not establish an individual's risk or a causal effect of latitude.")
else:
    print("Expected course files were not found. Keep using the self-contained examples above.")
""")
out=Path("STAT501_Lesson1_Simple_Linear_Regression_Python.ipynb")
nbf.write(nb,out)
print("Wrote",out,"with",len(nb.cells),"cells.")

