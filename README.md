# Business-Applications-for-Forecasting

_

### Prerequisites

_

### Installing

_

### Problem Defined

- **Definition of the problem**:
Forecast one day ahead for a wikipedia page
- **Goal**: Univariate time series forecasting on the first wikipedia page from on the train_1.csv file.  Deadline: Monday, September 17
- **Initial Pre-Model Pipeline**: Loading the train_1.csv file from the [Kaggle Competition](https://www.kaggle.com/c/web-traffic-time-series-forecasting) and
stationarize the time series.
- **Purpose**: The purpose of this project is to improve my ability to forecast
- **Example Business Use Cases** for forecasting are:
1. Customer data on a self-service platform
2. Identify anomalies for DevOps threat alerting (i.e. tracking is working; can save you millions by prevention)
3. Future time features for machine learning applications
4. Find upcoming costs and revenue for budgeting and borrowing money (i.e. lowering funding for teams)
5. Predict the price of a company resource by day/day of week/hour to purchase at the lowest cost (i.e. AWS spot instances)
6. Identifying the trend in the business to discover reasons to change to stay competitive (i.e. amount of business with each customer segment over time)
7. Spend more on marketing during a customer segment's peak purchasing day/day of week/hour
8. Learning what your customer segments are based on purchasing/web traffic day/day of week/hour
9. Forecasting key deliverables to help forecast revenue
10. Forecast recruitments/physical capital by department/team to help forecast cost
11. Forecast price of resources from suppliers used by the business by month to discover reasons to change to stay competitive
12. Keep track of key performance indicators of success for your business and competition to identify threats and change strategies (i.e. budget for improving the product in the short term, sales/marketing, hiring, or research and development) (Keep in mind how you stand out among the competition and how that has changed)
13. Keeping track of number of customer support data over time to find stories that help you change your tactics to address problem (i.e. time to resolution, time to first response, hiring to address high velocity of tickets, transferring employees to other teams if there is low velocity of tickets, number of high priority tickets)
14. Determining the need to open/close new warehouses
15. Determining when suppliers are trending towards are charging too much, or providing too little value.  Which times are the prices the lowest?
16. Finding the effect of different marketing efforts on closing sales

- There are hundreds of ways to add value to a business with forecasting.  Companies need to stay competitive and stand out among the competition to survive.  Use forecasting to keep an edge by adding value in one of the ways mentioned above.  As mentioned above, prevention of threatening events alone can save a company millions.  Fund a forecasting project today.

### Project Ideas

There are basically three often used approaches to make time series stable based on three difference scenarios: 1) first difference for linear trend; 2) log for non-linear trend; 3) log seasonal difference for seasonality.

Other methods are square, log difference, lag. However, sometime it is difficult to get stationary, then you could try different combinations of those techniques for example log square difference and so on. Stationarity is tested via unit root for example ADF, ADF-GLS, KPSS, PP, CH. If you can reject the null of unit root, you can confirm that the time series is stationary. There are three different models from ADF test: 1) no constant (not always used and be very cautious with this one); 2) with constant; 3) with constant and trend. You have to model ARIMA accordingly based on the test for example you must include a constant in ARIMA if you use the second and both a constant and a trend for the third. Please note that the second ADF model is commonly used, however wrong tests could reduce the power of ADF. Therefore, I also recommend you run several comparable tests to get a more robust result. According to Hacker (2010), the Schwarz information criterion can be helpful in determine which testing models to use in ADF.
