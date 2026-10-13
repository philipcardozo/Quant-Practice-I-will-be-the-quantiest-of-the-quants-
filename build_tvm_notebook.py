from pathlib import Path
import nbformat as nbf
nb=nbf.v4.new_notebook()
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}}
def md(x): nb.cells.append(nbf.v4.new_markdown_cell(x.strip()))
def c(x): nb.cells.append(nbf.v4.new_code_cell(x.strip()))

md(r"""
# Time Value of Money, Interest Rates, and Credit Risk
## A Python lab based on the supplied Saving and Borrowing lesson

This notebook covers the full supplied lesson: time value, interest rates, simple and compound interest, present/future value, nominal/effective/continuous compounding, yield curves, spot and forward rates, variable rates, real rates, saving versus borrowing, maturity, default risk, and applied cash-flow problems.

**Source:** the supplied WQU Financial Markets, Module 1: Credit Risk and Financing, Lesson 1: Saving and Borrowing notes. Course examples retain their stated inputs; plots are instructional reconstructions.
""")
md(r"""
## Roadmap

| Section | What you will be able to do |
|---|---|
| Time value and rates | Explain why a dollar today differs from a dollar later |
| PV/FV | Compound, discount, solve for rate and time |
| Compounding | Compare simple, nominal, effective, and continuous rates |
| Yield curve | Interpret maturity-specific rates and implied forwards |
| Variable / real rates | Model reset risk and purchasing-power return |
| Credit | Explain savings-loan spreads, maturity, and default-risk premiums |
| Applications | Value annuities and retirement cash flows |
| Interview practice | Solve five increasingly difficult analyst questions |
""")
c(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.figsize":(8,4.6),"axes.titleweight":"bold"})

def fv_simple(pv,r,t): return pv*(1+r*t)
def pv_simple(fv,r,t): return fv/(1+r*t)
def fv(pv,r,t): return pv*(1+r)**t
def pv(fv_value,r,t): return fv_value/(1+r)**t
def fv_nominal(pv_value,r,m,t): return pv_value*(1+r/m)**(m*t)
def ear(r,m): return (1+r/m)**m-1
def fv_continuous(pv_value,r,t): return pv_value*np.exp(r*t)
def pv_annuity(payment,r,n): return payment*(1-(1+r)**(-n))/r
def annuity_payment(pv_value,r,n): return pv_value*r/(1-(1+r)**(-n))
def forward(s_short,s_long,short_t,long_t):
    return ((1+s_long)**long_t/(1+s_short)**short_t)**(1/(long_t-short_t))-1
print("Formulas loaded.")
""")
md(r"""
---
# 1. Time value of money

A dollar today is worth more than a dollar received later: it can be consumed, saved, invested, or used to avoid debt now. Waiting deserves compensation. That compensation is the **interest rate** or required return.

- A saver/lender gives money today and receives principal plus interest later.
- A borrower receives money now and repays principal plus interest later.
- An investor trades money now for uncertain future cash flows.

Every calculation must specify its dates, currency, rate convention, and risk assumptions.
""")
md(r"""
# 2. Interest rates

Interest is the price of using someone else's money. A lender's required rate reflects time, expected inflation, liquidity, the borrower's default risk, and available alternatives.

**Course example:** lending 1,000 for one year at 5 percent produces 1,050 with annual compounding.
""")
c(r"""
print("Future value:",fv(1000,.05,1))
""")
md(r"""
---
# 3. Present and future values

## Simple interest

$$FV=PV(1+rt),\qquad PV={FV\over 1+rt}.$$

Interest is earned only on original principal.

## Compound interest

$$FV=PV(1+r)^t,\qquad PV={FV\over(1+r)^t}.$$

Interest earns interest. PV discounts future cash flows backwards; FV compounds current cash forward. Rate and time units must match.
""")
c(r"""
# Course example: 100,000 at 4.5 percent for 10 years.
principal,rate,years=100_000,.045,10
simple=fv_simple(principal,rate,years)
compound=fv(principal,rate,years)
display(pd.DataFrame({"method":["simple","annual compound"],"future value":[simple,compound],
                      "interest":[simple-principal,compound-principal]}).style.format({"future value":"USD {:,.2f}","interest":"USD {:,.2f}"}))
x=np.arange(31)
plt.plot(x,[fv_simple(principal,rate,z) for z in x],label="simple 4.5 percent",lw=2)
plt.plot(x,[fv(principal,rate,z) for z in x],label="compound 4.5 percent",lw=2)
plt.title("Compounding produces exponential growth");plt.xlabel("Years");plt.ylabel("Future value");plt.legend();plt.show()
""")
md(r"""
## Solve for the unknown rate or time

$$r=(FV/PV)^{1/t}-1,\qquad t={\ln(FV/PV)\over\ln(1+r)}.$$

Use these to convert an investment claim into an annual return or a time-to-goal. The course example of 1,000 becoming 1,250 in two years has an annual compound rate of about 11.8 percent.
""")
c(r"""
implied=(1250/1000)**.5-1
print("Implied annual rate:",format(implied,".2%"))
print("Exact years to double at 6 percent annually:",round(np.log(2)/np.log(1.06),2))
""")
md(r"""
## Nominal rates, effective annual rates, and continuous compounding

At nominal annual rate r compounded m times per year:

$$FV=PV(1+r/m)^{mt},\qquad EAR=(1+r/m)^m-1.$$

More frequent compounding raises the effective annual rate for the same quoted r. Compare like-for-like using EAR.

In the continuous limit:

$$FV=PVe^{rt},\qquad PV=FVe^{-rt}.$$

At 6 percent continuously compounded, doubling takes ln(2)/.06 = 11.55 years. The effective annual rate is $e^{.06}-1$.
""")
c(r"""
rows=[]
for label,m in [("annual",1),("semiannual",2),("quarterly",4),("monthly",12),("daily",365)]:
    rows.append([label,m,fv_nominal(10000,.05,m,1),ear(.05,m)])
display(pd.DataFrame(rows,columns=["frequency","m","FV of 10,000","EAR"]).style.format({"FV of 10,000":"USD {:,.2f}","EAR":"{:.5%}"}))
print("Course checks: monthly FV is about 10,511.62; quarterly is about 10,509.45.")
print("Continuous 6 percent doubling years:",round(np.log(2)/.06,4)," | EAR:",format(np.exp(.06)-1,".4%"))
""")
md(r"""
---
# 4. Yield curve

A **yield curve** plots rates by maturity for comparable instruments. It describes the term structure at one moment.

- An upward curve often compensates for a longer commitment, rate uncertainty, inflation uncertainty, and lower liquidity.
- A flat or inverted curve can occur when short rates are high or markets expect rate cuts/economic weakness.
- It is not a promise of future short rates.

Yield curves matter for bonds, mortgages, pension liabilities, project discount rates, and refinancing decisions.
""")
c(r"""
maturity=np.array([1/12,.25,.5,1,2,3,5,7,10,20,30])
up=np.array([.002,.003,.004,.006,.009,.015,.024,.031,.037,.044,.044])
inv=np.array([.052,.052,.052,.051,.049,.047,.045,.045,.046,.048,.047])
plt.plot(maturity,up,"o-",label="illustrative upward curve")
plt.plot(maturity,inv,"o-",label="illustrative inverted curve")
plt.gca().yaxis.set_major_formatter(lambda v,p:format(v,".0%"))
plt.title("Rates can vary across maturities");plt.xlabel("Maturity in years");plt.ylabel("Yield");plt.legend();plt.show()
""")
md(r"""
# 5. Spot and forward rates

A **spot rate** is today's annualized rate from now to a specific maturity. A **forward rate** is today's implied rate for a future interval.

$$f(t,t+n)=\left[{(1+s_{t+n})^{t+n}\over(1+s_t)^t}\right]^{1/n}-1.$$

No-arbitrage intuition: investing long today must equal investing short today then reinvesting at the implied forward. A forward rate is a market-implied price, not a guaranteed forecast.
""")
c(r"""
s1,s2=.055,.0602
f12=forward(s1,s2,1,2)
print("one-year spot:",format(s1,".2%"),"two-year spot:",format(s2,".2%"))
print("one-year forward beginning in one year:",format(f12,".3%"))
print("two-year growth:",(1+s2)**2,"short then forward:",(1+s1)*(1+f12))
""")
md(r"""
---
# 6. Variable interest rates

Fixed-rate borrowing has predictable payments. A variable-rate loan resets with a benchmark plus spread, so payment and interest expense can change. The course uses a credit-card-style example with a changing periodic rate.

Assess expected cost **and** affordability after shocks. Do not choose fixed or floating solely from a rate forecast: model reset scenarios, refinancing risk, duration of need, and cash-flow resilience.
""")
c(r"""
balance=2500
monthly=np.array([.024,.024,.025,.030,.038,.040])
display(pd.DataFrame({"month":np.arange(1,7),"monthly rate":monthly,"interest on 2,500":balance*monthly}).style.format({"monthly rate":"{:.2%}","interest on 2,500":"USD {:,.2f}"}))
""")
md(r"""
---
# 7. Nominal versus real rates

Nominal returns measure money growth. Real returns measure purchasing-power growth.

Approximation:

$$r_{real}\approx r_{nominal}-inflation.$$

Exact Fisher relation:

$$(1+r_{nominal})=(1+r_{real})(1+inflation),\qquad r_{real}={1+r_{nominal}\over1+inflation}-1.$$

A positive nominal yield can be a negative real yield.
""")
c(r"""
cases=pd.DataFrame({"nominal":[.03,.02,.08],"inflation":[.02,.05,.03]})
cases["approx real"]=cases.nominal-cases.inflation
cases["exact real"]=(1+cases.nominal)/(1+cases.inflation)-1
display(cases.style.format("{:.2%}"))
""")
md(r"""
---
# 8. Rate of saving versus rate of borrowing

Banks typically pay less on deposits than they charge on loans. The spread funds operating costs, expected credit losses, capital/liquidity costs, and profit.

**Course example:** 10,000 at 3.3 percent earns 330 in one year; a 10,000 loan at 7.5 percent costs 750. Compare products, but understand that the spread also reflects the difference between a safe deposit liability and risky lending asset.
""")
c(r"""
print("Deposit interest at 3.3 percent:",10000*.033)
print("Loan interest at 7.5 percent:",10000*.075)
print("Difference:",10000*(.075-.033))
""")
md(r"""
# 9. Term to maturity, 10. different maturities, and 11. default risk

**Term to maturity** is remaining time until principal repayment. Longer lockups often require a premium because investors lose flexibility and bear more rate/inflation uncertainty. The actual relationship across maturities is summarized by the yield curve.

**Default risk** is the risk that a borrower cannot or will not repay. Lower credit quality normally requires a higher promised yield. A high promised yield is compensation for risk, not a free return; expected defaults and recovery matter.
""")
c(r"""
principal=10000
safe_fv=principal*1.04
risky_fv=(1-.08)*principal*1.09+.08*principal*.40
display(pd.DataFrame({"investment":["safe bond","risky bond"],"promised FV":[safe_fv,principal*1.09],
                      "expected FV":[safe_fv,risky_fv]}).style.format({"promised FV":"USD {:,.2f}","expected FV":"USD {:,.2f}"}))
print("Expected value is a simple illustration, not a full credit-pricing model.")
""")
md(r"""
---
# 12. Applied problems: annuities, retirement, loans

A TVM problem is a timeline problem. Draw dates, signs, and rates; select a focal date; compound earlier cash flows forward and discount later ones back.

For an ordinary annuity with end-of-period payment C:

$$PV=C{1-(1+r)^{-n}\over r},\qquad C=PV{r\over1-(1+r)^{-n}}.$$

**Illustrative retirement calculation:** 1,000,000 compounded at 3 percent for 40 years becomes about 3.26 million. The following cell then asks what 20-year year-end withdrawal that balance can support at 5 percent.
""")
c(r"""
retirement=fv(1_000_000,.03,40)
withdraw=annuity_payment(retirement,.05,20)
print("Retirement balance:",format(retirement,",.0f"))
print("Level year-end withdrawal:",format(withdraw,",.2f"))
bal=retirement;rows=[]
for yr in range(1,4):
    interest=bal*.05;end=bal+interest-withdraw
    rows.append([yr,bal,interest,withdraw,end]);bal=end
display(pd.DataFrame(rows,columns=["year","opening","interest","withdrawal","ending"]).style.format("USD {:,.2f}"))
""")
md(r"""
## Decision-ready TVM checklist

1. Draw the timeline and identify inflows/outflows.
2. Name the rate convention: simple, EAR, nominal with m periods, or continuous.
3. Align periods: monthly cash flows require monthly rates.
4. Check scope: higher r or longer t should increase FV and decrease PV.
5. Separate nominal from real outcomes.
6. Explain risks: inflation, liquidity, default, and reset risk.
7. Report sensitivity, not a single false-precision answer.
""")
md(r"""
---
# Interview practice: five progressively harder questions

A high-quality interview response gives the calculation, interprets it in economic terms, and states a caveat or next analysis.
""")
md(r"""
## 1. Compound a deposit

**Prompt:** Deposit 5,000 at 6 percent annually for three years. What is the future value, and why is it not simply 5,900?
""")
c(r"""
print("Future value:",format(fv(5000,.06,3),",.2f"))
print("It exceeds 5,900 because years two and three earn interest on prior interest.")
""")
md(r"""
## 2. Compare savings offers

**Prompt:** Bank A quotes 6 percent compounded monthly. Bank B quotes 6.1 percent compounded annually. Which pays more over one year?
""")
c(r"""
a,b=ear(.06,12),.061
print("A EAR:",format(a,".4%"),"B EAR:",format(b,".4%"))
print("A pays more for risk-equivalent one-year funds because monthly compounding lifts its EAR above 6.1 percent.")
print("Also check fees, insurance, withdrawal restrictions, and reset terms.")
""")
md(r"""
## 3. Explain a real return

**Prompt:** A portfolio earns 4 percent when inflation is 5 percent. Find the real return and describe the approximation.
""")
c(r"""
print("Exact real:",format(1.04/1.05-1,".3%"),"Approximation:",format(.04-.05,".2%"))
print("Use the exact Fisher relation when precision or rate levels matter.")
""")
md(r"""
## 4. Derive a forward rate

**Prompt:** One-year and two-year spots are 4 percent and 5 percent. Find the one-year forward rate starting next year. What does it mean?
""")
c(r"""
ans=forward(.04,.05,1,2)
print("Forward f(1,2):",format(ans,".4%"))
print("It equalizes a two-year investment with a one-year investment rolled over for year two. It is implied pricing, not a certainty.")
""")
md(r"""
## 5. Credit-and-cash-flow recommendation

**Prompt:** A client can borrow 200,000 at floating reference plus 2 percent, currently 5.5 percent, or fixed 6.2 percent for five years. Cash flow is tight. How do you advise?
""")
c(r"""
print("\n".join([
"1. Current floating cost is lower, but price alone is not the decision.",
"2. Stress-test payments at +1, +2, and +3 percentage points against cash-flow capacity.",
"3. Review refinancing/prepayment terms, income sensitivity to rates, and loan horizon.",
"4. If a rate shock threatens solvency, fixed-rate certainty can be valuable insurance.",
"5. Present break-even costs and a recommendation tied to risk tolerance, not a rate forecast."
]))
""")
md(r"""
---
# Takeaway sheet

- PV discounts and FV compounds; timing and compounding convention drive the answer.
- Compare nominal quotes through effective rates; continuous rates require exponential conversion.
- Yield curves describe maturity-specific rates; forwards are implied no-arbitrage prices.
- Real return measures purchasing power, not just money growth.
- Credit spreads compensate for maturity, liquidity, and default risk.
- A professional TVM answer makes its dates, rates, risk assumptions, and sensitivities explicit.
""")
out=Path("Credit Risk/TimeValue of Money/Time_Value_of_Money_Saving_Borrowing_Python.ipynb")
nbf.write(nb,out)
print("Wrote",out,"with",len(nb.cells),"cells.")

