import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics
from sklearn.preprocessing import StandardScaler


def inspect_data(df):
    # 1. Display the first few rows of the data
    print("First 5 rows of the dataset:")
    print(df.head())

    # 2. Display the structure of the dataset
    print("\nDataset structure (columns and data types):")
    print(df.info())

    # 3. Display summary statistics of numeric columns
    print("\nSummary statistics:")
    print(df.describe())

    # 4. Show column names
    print("\nColumn names:")
    print(df.columns.tolist())

    # 5. Display the number of missing values per column
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # 6. Show dataset shape (number of rows and columns)
    print("\nDataset shape:")
    print(df.shape)

    # 7. Display unique values in categorical columns (optional)
    categorical_cols = df.select_dtypes(include=['object']).columns
    print("\nUnique values in categorical columns:")
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values")


def clean_data(df):
    # 删除任何包含缺失值的行
    df_cleaned = df.dropna()
    # 删除包含缺失值的列
    df_cleaned = df.dropna(axis=1)

    # 使用列的均值填充缺失值
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    # 使用指定值填充缺失值
    df['Salary'] = df['Salary'].fillna('Unknown')

    # 删除完全相同的重复行
    df_cleaned = df.drop_duplicates()
    # 删除某些列的重复行（仅根据某些列来判断重复）
    df_cleaned = df.drop_duplicates(subset=['Name', 'Age'])

    # 将'Age'列转换为整数类型
    df['Age'] = df['Age'].astype(int)
    # 将'Date'列转换为日期格式
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    # 提取年、月、日
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    # 时间序列重采样 df.resample('M').mean()

    # 假设我们要将Salary大于100000的值修正为100000
    df['Salary'] = df['Salary'].clip(upper=100000)
    # 删除'Age'列中小于20或大于60的值
    df = df[(df['Age'] >= 20) & (df['Age'] <= 60)]

    # 将'Gender'列中的'Male'替换为'M'
    df['Gender'] = df['Gender'].replace('Male', 'M')
    # 将多个值同时替换
    df['Gender'] = df['Gender'].replace({'Male': 'M', 'Female': 'F'})

    df_cleaned = df.resample('W').mean()  # 时间序列重采样，参数'W'把日线转为周线

    return df_cleaned


def select_data(df):
    # 使用 .loc[] 选择指定行和列的单个元素
    print(df.loc[1, 'Name'])  # 获取第二行（标签为1）'Name'列的值
    # 使用 .iloc[] 选择第二行（位置1）和第一列（位置0）的元素
    print(df.iloc[1, 0])  # 获取第二行，第一列的值
    # 选择多行多列，第1行到第2行（标签0到1）和'Name'与'Age'两列
    print(df.loc[0:1, ['Name', 'Age']])
    # 选择第1行到第2行（位置0到1）和第1列到第2列（位置0到1）
    print(df.iloc[0:2, 0:2])

    # 使用 df[] 进行筛选
    # 使用 df[] 访问 'Column 1' 列
    print(df['Name'])
    # 条件筛选：选择 Age > 30 的行
    print(df[df['Age'] > 30])
    # 使用 &（与）和 |（或）运算符可以组合多个条件帅选
    print(df[(df['Age'] > 30) & (df['Salary'] > 60000)])


def fun_data(df):
    # 对 'Age' 列和 'Salary' 列进行操作（如果当前列的名称是 'Age'，则对该列的所有值执行 x * 2 操作。
    # else x * 1.1：如果当前列的名称是 'Salary'，则对该列的所有值执行 x * 1.1 操作）
    df[['Age', 'Salary']] = df[['Age', 'Salary']].apply(lambda x: x * 2 if x.name == 'Age' else x * 1.1)

    print(df)


def feature(df):
    # 特征工程
    # 时间序列
    # df['year'] = df['date'].dt.year
    # df['month'] = df['date'].dt.month
    # df['day'] = df['date'].dt.day

    # one-hot编码
    df = pd.get_dummies(df, columns=['Gender'])

    # 数据均一化
    scaler = StandardScaler()
    df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])


def mod_train():
    print('machine learning...')


def visualize(df):
    df.plot(figsize=(10, 12), subplots=True)  # subplots参数将分图表展示
    plt.show()


if __name__ == '__main__':
    data = {'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [30, 40, 25],
            'Salary': [50000, 80000, 45000],
            'Date': ['2010.6.30', '2010.6.30', '2010.6.30'],
            'Gender': ['Female', 'Female', 'Female']}
    df = pd.DataFrame(data)
    # # Load your data into a DataFrame
    # df = pd.read_csv('./your_file.csv')
    inspect_data(df)
    clean_data(df)
    select_data(df)
    feature(df)
    mod_train()
    visualize(df)
