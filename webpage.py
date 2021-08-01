import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import chart_studio.plotly as py
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pycountry as pc


@st.cache
def load_data():
    cnx = sqlite3.connect('database.db')
    train_sql_table = pd.read_sql_query("SELECT * FROM train", cnx)
    cnx.close()
    return train_sql_table

def getCount(series):
    count = series.value_counts()
    xValues = count.index
    yPercent = (count/count.sum())*100
    return xValues, yPercent


st.title('HOTEL DEMAND BOOKING')

st.sidebar.title('DISPLAY FEATURES')

check_box = st.sidebar.checkbox(label="Display dataset")

train_data = load_data()

if check_box:
    st.write(train_data.head(10000))

view=0
hotel_but=st.sidebar.button('Hotel',key=1)
year_but=st.sidebar.button('Year',key=1)
month_but=st.sidebar.button('Month',key=1)
country_but=st.sidebar.button('Country',key=1)
accom_but=st.sidebar.button('Accommodation',key=1)
marseg_but=st.sidebar.button('marketSegment',key=1)
custyp_but=st.sidebar.button('CustomerType',key=1)
other_but=st.sidebar.button('Others',key=1)
#view = st.sidebar.radio(label="View By",options=['Hotel','Year','Month','Country','Accommodation','marketSegment','customerType','Other'])

if hotel_but == 1:
    st.subheader('How many bookings got cancelled')
    xCancelled, yCancelled = getCount(train_data["is_canceled"])
    plotly_figure = px.bar(data_frame=train_data,x=xCancelled,y=yCancelled,title='Booking Cancellation ratio')
    st.plotly_chart(plotly_figure)

    st.subheader('How many bookings got placed')
    xBookingByHotel, yBookingByHotel = getCount(train_data['hotel'])
    plotly_figure = px.bar(data_frame=train_data,x=xBookingByHotel,y=yBookingByHotel, title="Booking ratio (By Hotel)")
    st.plotly_chart(plotly_figure)

    #doubtful
    # df_1='hotel'
    # df_2='is_canceled'
    # plotly_figure = px.bar(data_frame=train_data,x=train_data[df_1],y=train_data[df_2], title="Booking Cancellation ratio (By Hotel)")
    # st.plotly_chart(plotly_figure)


    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('Cancelation Ratio of City hotel and Resort hotel')
    plt.subplots(figsize=(12,8))
    sns.countplot(x='hotel',hue='is_canceled',data=train_data)
    st.pyplot()

    st.subheader('Duration of stay (by hotel type)')
    train_new = train_data.copy()
    train_new['arrival_date_comb']= train_new['arrival_date_month'] + train_new['arrival_date_year'].astype(str)
    train_new['total_stay'] = train_new['stays_in_week_nights'] + train_new['stays_in_weekend_nights']
    plt.subplots(figsize=(15,10))
    sns.countplot(x='total_stay',hue='hotel',data=train_new, order=train_new.total_stay.value_counts().iloc[:11].index)
    st.pyplot()




if year_but==1:
    st.subheader('Booking with respect to the years')
    xYear, yYear = getCount(train_data['arrival_date_year'])
    plotly_figure = px.bar(data_frame=train_data,x=xYear,y=yYear, title="Booking rate (by year)")
    st.plotly_chart(plotly_figure)

    st.subheader('Booking cancellation with respect to the years')
    plt.subplots(figsize=(8,6))
    sns.countplot(x='arrival_date_year',hue='is_canceled',data=train_data)
    st.pyplot()


if month_but== 1:
    st.subheader('Booking cancellation with respect to the months')
    train_new = train_data.copy()
    train_new['arrival_date_comb']= train_new['arrival_date_month'] + train_new['arrival_date_year'].astype(str)
    plt.subplots(figsize=(30,15))
    sns.countplot(x='arrival_date_comb',hue='is_canceled',data=train_new)
    st.pyplot()

    st.subheader('Respect to months')
    plt.subplots(figsize=(12,8))
    sns.countplot(x='arrival_date_month',hue='is_canceled',data=train_data)
    st.pyplot()


    st.subheader('Busiest month')
    month = ['January','February','March','April','May','June','June','July','August','September','October','November','December']
    data_non_cancelled = train_data[train_data['is_canceled']==0]
    sorted_data = data_non_cancelled['arrival_date_month'].value_counts().reindex(month)
    xMonths = sorted_data.index
    yMonths = (sorted_data/sorted_data.sum())*100
    plotly_figure = px.line(data_frame=sorted_data,x=xMonths ,y=yMonths.values, title="Booking by Month")
    st.plotly_chart(plotly_figure)

    st.subheader('Busiest month (by hotel type)')
    sorted_data = data_non_cancelled.loc[train_data['hotel']=='City Hotel','arrival_date_month'].value_counts().reindex(month)
    xMonthsCity = sorted_data.index
    yMonthsCity = (sorted_data/sorted_data.sum())*100
    sorted_data = data_non_cancelled.loc[train_data['hotel']=='Resort Hotel','arrival_date_month'].value_counts().reindex(month)
    xMonthsResort = sorted_data.index
    yMonthsResort = (sorted_data/sorted_data.sum())*100
    fig,axes = plt.subplots(figsize=(14,6))
    axes.set_xlabel('Months')
    axes.set_ylabel('Percentage % of occupancy')
    axes.set_title('Booking By Month (Hotel Type)')
    sns.lineplot(x=xMonthsCity,y=yMonthsCity,label='City Hotel',sort=False)
    sns.lineplot(x=xMonthsResort,y=yMonthsResort,label='Resort Hotel',sort=False)
    st.pyplot()

if country_but == 1:
    xCustomers, yCustomers = getCount(train_data['country'])
    xCustomers = xCustomers[:][:11]
    yCustomers = yCustomers[:][:11]
    country_name = [pc.countries.get(alpha_3=name) for name in xCustomers]
    def isNone(data):
        if(data==None):
            return True
        else:
            return False
    country_name = [x for x in country_name if not isNone(x)]
    country_name = [country_name[i].name for i in range(0,len(country_name))]
    # myPlot(country_name,yCustomers,"Countries","% Booking","Booking by country", (15,15),'bar')
    plotly_figure = px.bar(data_frame=country_name,x=xCustomers ,y=yCustomers, title="Booking by country")
    st.plotly_chart(plotly_figure)


if accom_but == 1:
    totalStay = train_data['stays_in_week_nights'] + train_data['stays_in_weekend_nights']
    xStay, yStay = getCount(totalStay)
    xStay = xStay[:][0:11]
    yStay = yStay[:][0:11]
    # myPlot(xStay,yStay,"Number of nights","Percentage %","Duration of Stay",(15,15),'bar')
    plotly_figure = px.bar(data_frame=totalStay,x=xStay ,y=yStay, title="Duration of Stay")
    st.plotly_chart(plotly_figure)


    single = train_data[(train_data.adults==1) & (train_data.children==0) & (train_data.babies==0)]
    couple = train_data[(train_data.adults==2) & (train_data.children==0) & (train_data.babies==0)]
    family = train_data[train_data.adults+train_data.children+train_data.babies > 2]
    label = ['Single', 'Couple', 'Family']
    count = [single.shape[0],couple.shape[0],family.shape[0]]
    countPercent = [(i/train_data.shape[0])*100 for i in count]
    # myPlot(label, countPercent,"Type of accomodation","Percentage %","Booking by Accomodation",(6,10),'bar')
    plotly_figure = px.bar(x= label,y=countPercent, title="Booking by Accomodation")
    st.plotly_chart(plotly_figure)


if marseg_but == 1:
    xBookingBySegment, yBookingBySegment = getCount(train_data['market_segment'])
    xBookingBySegment = xBookingBySegment[:][:-1]
    yBookingBySegment = yBookingBySegment[:][:-1]
    # myPlot(xBookingBySegment, xBookingBySegment, xLable="Market Segment", yLable="% Percentage", title="Booking ratio (by market segment)",figsize=(8,8),type='bar')
    plotly_figure = px.bar(x= xBookingBySegment,y=yBookingBySegment, title="Booking ratio (by market segment)")
    st.plotly_chart(plotly_figure)


    st.subheader('Cancellation by Market Segment')
    plt.subplots(figsize=(14,8))
    sns.countplot(x='market_segment',hue='is_canceled',data=train_data)
    st.pyplot()

    xBookingCustType, yBookingCustType = getCount(train_data['customer_type'])
    # myPlot(xBookingCustType, yBookingCustType, xLable="Customer Type", yLable="% Percentage", title="Booking ratio (by customer type)",figsize=(8,8),type='bar')
    plotly_figure = px.bar(x= xBookingCustType,y=yBookingCustType, title="Booking ratio (by customer type)")
    st.plotly_chart(plotly_figure)

if custyp_but == 1:
    st.subheader('Cancellations by customer type')
    plt.subplots(figsize=(14,8))
    sns.countplot(x='customer_type',hue='is_canceled',data=train_data)
    st.pyplot()

    st.subheader('Cancellations by customer with deposit type')
    plt.subplots(figsize=(14,8))
    sns.countplot(x='deposit_type',hue='is_canceled',data=train_data)
    st.pyplot()



    st.subheader('Customer Type')
    plt.figure(figsize=(16,12))
    plt.subplot(221)
    sns.countplot(x=train_data['meal'], hue=train_data['is_canceled'])
    plt.xlabel('Meal Type')
    plt.subplot(222)
    sns.countplot(x=train_data['customer_type'], hue=train_data['is_canceled'])
    plt.xlabel('Customer Type')
    plt.subplot(223)
    sns.countplot(x=train_data['reserved_room_type'], hue=train_data['is_canceled'])
    plt.xlabel('Reserved Room Type')
    plt.subplot(224)
    sns.countplot(x=train_data['reservation_status'], hue=train_data['is_canceled'])
    plt.xlabel('Reservation Status')
    plt.show()
    st.pyplot()

if other_but == 1:
    st.subheader('Other')
    train_data['arrival_date'] = train_data['arrival_date_year'].astype(str) + '-' + train_data['arrival_date_month'] + '-' + train_data['arrival_date_day_of_month'].astype(str)
    train_data['arrival_date'] = train_data['arrival_date'].apply(pd.to_datetime)
    train_data['reservation_status_date'] = train_data['reservation_status_date'].apply(pd.to_datetime)
    cancelled_data = train_data[train_data['reservation_status'] == 'Canceled']
    cancelled_data['canc_to_arrival_days'] = cancelled_data['arrival_date'] - cancelled_data['reservation_status_date']
    cancelled_data['canc_to_arrival_days'] = cancelled_data['canc_to_arrival_days'].dt.days
    plt.figure(figsize=(14,6))
    sns.histplot(x=cancelled_data['canc_to_arrival_days'])
    plt.show()
    st.pyplot()



    st.write('Percentage of cancellations that are within a week of arrival: ', 
        (cancelled_data[cancelled_data['canc_to_arrival_days']<=7]['canc_to_arrival_days'].count()*100/cancelled_data['canc_to_arrival_days'].count()).round(2), '%')


    plt.figure(figsize=(8,8))
    explode = [0.005] * len(cancelled_data['market_segment'].unique())
    colors = ['royalblue','orange','y','darkgreen','gray','purple','red','lightblue']
    plt.pie(cancelled_data['market_segment'].value_counts(),autopct = '%.1f%%',explode = explode,colors = colors)
    plt.legend(cancelled_data['market_segment'].unique(), bbox_to_anchor=(-0.1, 1.),fontsize=14)
    plt.title('Market Segment vs Cancelled Bookings')
    plt.tight_layout()
    plt.show()
    st.pyplot()


    plt.figure(figsize=(12,8))
    train_data.corr()['is_canceled'].sort_values()[:-1].plot(kind='bar')
    plt.show()
    st.pyplot()