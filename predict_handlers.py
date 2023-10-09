# ---------------------- Define schemas of Data -------------------------
from typing import List, Optional, Any
from pydantic import BaseModel
import pandas as pd

# # #Tạo thư viện sử dụng
def normalize_API(df_full, path_df, parameter):
    import sys
    sys.path.append('../')
    #Nhập các thư viện cần thiết, để xây dựng thuật toán
    import lasio
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import math
    import seaborn as sns
    import numpy as np
    from scipy import stats
    from sklearn.preprocessing import MinMaxScaler
    from scipy.cluster.vq import kmeans, vq
    from sklearn.preprocessing import StandardScaler
    # import tensorflow as tf
    # from tensorflow.keras import layers
    from sklearn.mixture import GaussianMixture

#     #Setup parameter for API
    top = parameter.get("top", []) #Tên cột nóc hệ tầng
    base = parameter.get("base", []) #Tên cột đáy hệ tầng
    wells_to_remove = parameter.get("wells_to_remove", []) #Giếng bị loại bỏ do khác biệt phân bố dữ liệu
    threshold_outside_range = parameter.get("threshold_outside_range", [])
    algorithm = parameter.get("algorithm") #algorithm #"Auto selection' #'Manual'
    well = parameter.get("well", []) #Lựa chọn tên giếng theo phương pháp thủ công
    linear_shift = parameter.get("linear_shift")

    #Lựa chọn phương pháp chuẩn hoá theo giá trị 0-1
    MinMax_scale = parameter.get("MinMax_scale") #On là theo kết quả chuẩn hoá sẽ được trả đầu ra là tỷ lệ thang đo 0-1, còn Off theo tỷ lệ thang đo mặc định đường
    
    #Lựa chọn áp dụng zscore để phân cụm giếng
    zscore = parameter.get("zscore") #On là sử  dụng, còn Off là không sử dụng

    #Chọn lọc dữ liệu theo khoảng chiều sâu và hệ tầng (Bắt buộc người dùng nhập phải nhập liệu)
    filter_data = parameter.get("filter_data") #Lựa chọn lọc theo hệ tầng, nếu thay Formation = depth thì sẽ lọc theo giá trị depth, còn nếu chọn "both" thì chọn cả 2 tính năng chọn lọc.
    formation_column = parameter.get("formation_column", []) #Tên cột chứ các hệ tầng
    formation_name = parameter.get("formation_name", []) #Tên hệ tầng có trong cột chứa 
    formation = parameter.get("Formation", []) #Biến gán hệ tầng
    depth_column = parameter.get("depth_column") #Tên cột Depth
    min_depth = parameter.get("min_depth") #Người dùng nhập liệu giá trị độ sâu nhỏ nhất muốn lọc
    max_depth = parameter.get("max_depth") #Người dùng nhập liệu giá trị độ sâu lớn nhất muốn lọc

    #Chọn lọc dữ liệu theo chất lượng đo ghi, dựa trên đường DCALI
    upper_value_DCAL = parameter.get("upper_value_DCAL") #Người dùng nhập liệu giá trị nhỏ nhất muốn giữ lại dữ liệu (theo tính năng người dùng tự nhập liệu)
    lower_value_DCAL = parameter.get("lower_value_DCAL") #Người dùng nhập liệu giá trị lớn nhất muốn giữ lại dữ liệu (theo tính năng người dùng tự nhập liệu)

    #Chọn lọc và lựa chọn dữ liệu cho quá trình chuẩn hoá 
    min_quantile_value = parameter.get("min_quantile_value") #Người dùng nhập liệu giá trị xác xuất nhỏ nhất muốn xác định (theo tính năng người dùng tự nhập liệu)
    max_quantile_value = parameter.get("max_quantile_value") #Người dùng nhập liệu giá trị xác xuất lớn nhất muốn xác định (theo tính năng người dùng tự nhập liệu)
    ref_low = parameter.get("ref_low") #Người dùng nhập liệu giá trị xác xuất lớn nhất muốn xác định (theo tính năng người dùng tự nhập liệu)
    ref_high = parameter.get("ref_high") #Người dùng nhập liệu giá trị xác xuất nhỏ nhất muốn xác định (theo tính năng người dùng tự nhập liệu)
    curve = parameter.get("curve") #Người dùng nhập liệu tên đường cong muốn thực hiện việc chuẩn hoá (bắt buộc phải nhập liệu)
    a = parameter.get("a") #Tham số áp dụng cho quá trình chuẩn hoá một điểm
    zscore_threshold = parameter.get("zscore_threshold") #Ngưỡng xác định điểm dị thường

    #Người dùng xác định đường cong sử dụng để chuẩn hoá
    # column_name = parameter.get("column_name") #Tên cột đường cong muốn chuẩn hoá
    curve_name =  parameter.get("curve_name") #Tên đường cong muốn chuẩn hoá
    normalized_curve_name = parameter.get("normalized_curve_name") #Tên đường cong sau khi chuẩn hoá
    removed_well =  parameter.get("removed_well")
    remain_well =  parameter.get("remain_well")

    #Danh sách giếng khoan được nhập vào
    def well_list(df):
        '''Hàm này dùng để in ra danh sách tên các giếng khoan đã được nhập vào'''
        num_unique_values = df['WELL'].nunique()
        print ('Số giếng khoan:', num_unique_values)
        return print ('Danh sách giếng khoan:', df['WELL'].unique())

    #Thống kê số điểm có data và null data cho từng cột
    def data_info (df):
        """Hàm này dùng để thống kê số điểm có giá trị và 
        số điềm không có giá trị (null data) của tất cả các cột trong dataframe"""
        print ('Thống kê số điểm dữ liệu của từng cột')
        display(df.info())
        print(df.isnull().sum())
        print ('Thống kê số điểm không có dữ liệu của từng cột')
        missing_values_df = pd.DataFrame(df.isnull().sum(), columns=['missing_values_count'])
        display(missing_values_df)
        subset_cols = missing_values_df[missing_values_df['missing_values_count'] > 0].index.tolist()
        print(subset_cols)
        return subset_cols
    #Loại bỏ các điểm null data trong đường log
    def drop_na_subset(df, curve_name):
        """
        Hàm này loại bỏ các hàng có giá trị NaN trong các cột con trong DataFrame df.
        Arguments:
        df -- DataFrame đầu vào
        subset_cols -- Danh sách tên các cột con để kiểm tra giá trị NaN  
        Returns:
        DataFrame -- DataFrame mới đã loại bỏ các hàng có giá trị NaN
        """
        return df.dropna(subset=curve_name)
    

    #Hàm loại bỏ giếng khoan theo yêu cầu người dùng    
    def calculate_percentage_outside_range(data, curve_name, threshold_outside_range):
        min = data[curve_name].quantile(float(min_quantile_value))
        max = data[curve_name].quantile(float(max_quantile_value))

        print(f"Giá trị tập hợp điểm đường {curve_name} của tất cả các giếng sử dụng")
        print(f"P{float(min_quantile_value)*100:.0f}: {min:.2f}, P{float(max_quantile_value)*100:.0f}: {max:.2f}")
        
        data_list_2 = []
        for well in data['WELL'].unique():
            data_list_2 = data[data['WELL'] == well]
            min_ = data_list_2[curve_name].quantile(float(min_quantile_value))
            max_ = data_list_2[curve_name].quantile(float(max_quantile_value))
            print(f"Giá trị phân vị phần tử (quantile) tập hợp điểm của giếng {well}")
            print(f"P{float(min_quantile_value)*100:.0f}: {min_:.2f}, P{float(max_quantile_value)*100:.0f}: {max_:.2f}")

        data_list_1 = []
        for well in data['WELL'].unique():
            well_data = data[data['WELL'] == well]

            count_On_max = (well_data[curve_name] > max).sum()
            count_On_min = (well_data[curve_name] < min).sum()

            percentage_outside_range = round((count_On_max + count_On_min) / len(well_data) * 100, 2)

            print(f"Phần trăm số điểm nằm ngoài khoảng từ P{float(min_quantile_value)*100:.0f} - P{float(max_quantile_value)*100:.0f} của giếng khoan {well}: {percentage_outside_range}%")

            if percentage_outside_range <= float(threshold_outside_range):
                data_list_1.append({
                    'Well': well,
                    'Percentage Outside Range': percentage_outside_range
                })
            else:
                print(f"Giếng khoan {well} bị loại ra vì percentage_outside_range > {float(threshold_outside_range)}%")
        
        excluded_wells = data[data['WELL'].isin([item['Well'] for item in data_list_1])]
        return excluded_wells
    #Hàm loại bỏ giếng khoan theo yêu cầu người dùng 
    def remove_well(df, wells_to_remove):
        '''Hàm này dùng để loại bỏ giếng khoan bất kỳ
        Sử dụng toán tử ~ trước điều kiện này, chúng ta lấy bản ghi mà không khớp 
        với danh sách giếng khoan và trả về DataFrame chỉ chứa các giếng không bị loại bỏ '''
        return df[~df['WELL'].isin(wells_to_remove)]

    #Nhập bảng marker và kết hợp với dữ liệu log
    def marker_table(path_df,top, base, well, formation):
        # df = remove_well(df, wells_to_remove)
        try:
            df_formation = pd.DataFrame()
            df= path_df
            df_formation['WELL'] = df[[well]].copy().astype(str)
            df_formation['Formation'] = df[formation].copy().astype(str)
            df_formation['Top_depth'] = df[top].copy().astype(float)
            df_formation['Base_depth'] = df[base].copy().astype(float)
        except Exception as e:
            # print("An error occurred:", e)
            print("Done marker_table")
        return (df_formation)
    #Combine marker with well log values
    def add_formations_to_df(row):
        df_formation = marker_table(path_df,'Top_depth', 'Base_depth','WELL', 'Formation')
        well = row['WELL']
        depth_value = row['DEPTH']
        df = df_formation[df_formation['WELL'] == well]
        for _, formation_row in df.iterrows():
            if depth_value >= formation_row['Top_depth'] and depth_value <= formation_row['Base_depth']:
                return formation_row['Formation']
        return ''
        

    #Lựa chọn khoảng nghiên cứu theo hệ tầng
    def filtered_formation (df, formation_column, formation_name):
        '''Hàm này dùng để lọc dữ liệu theo hệ tầng'''
        df['Formation'] = df.apply(add_formations_to_df, axis=1)     
        filtered_df = df.loc[df["Formation"] == formation_name]
        print ('Hệ tầng được lựa chọn:', formation_name)
        return filtered_df

    #Lựa chọn khoảng nghiên cứu theo độ sâu
    def filtered_depth (df, depth_column, min_depth, max_depth):
        df['Formation'] = df.apply(add_formations_to_df, axis=1)
        '''Hàm này dùng để lọc dữ liệu trong một khoảng độ sâu tuỳ chọn'''
        filtered_df = df.loc[(df[depth_column] >= min_depth) & (df[depth_column] <= max_depth)]
        print ('Khoảng độ sâu được lựa chọn:', 'từ', min_depth,'m', 'đến', max_depth,'m')
        return filtered_df

    #Phương pháp z-score xác định điểm outlier
    def mark_outliers_zscore(curve_name, zscore_threshold):
        threshold_ = float(zscore_threshold)
        # get the z score
        z = np.abs(stats.zscore(curve_name))
        # return marked value if above threshold 
        return  [1 if value >= threshold_  else 0 for value in z]

    def process_zscore_data(df, curve_name, zscore_threshold):
        '''Hàm này dùng để tính số điểm outlier cho đường log dựa trên phương pháp z-score'''
        threshold_ = float(zscore_threshold)
        data_list = []
        for well in df.WELL.unique():
            data = df.loc[df['WELL'] == well]
            res = mark_outliers_zscore(data[curve_name], threshold_)
            print(well, ": Number of outliers: ", np.sum(res))
            data["Outlier_" + curve_name] = res
            data_list.append(data)
        
        df2 = pd.concat(data_list, axis=0)
        return df2
        
    def remove_outliers(df):
        '''Hàm này dùng để tạo dataframe với log curve đã được loại bỏ các điểm outlier xác định bằng phương pháp zscore ở trên'''
        return df[df["Outlier_GR"] != 1]

    #Lựa chọn khoảng có thành giếng khoan tốt không bị sập lở dựa vào đường DCAL
    def on_gauge_hole (df, upper_value_DCAL, lower_value_DCAL):
        '''Hàm này dùng để lọc dữ liệu thuộc khoảng thành giếng khoan không bị sập lở'''
        upper = float(upper_value_DCAL)
        lower = float(lower_value_DCAL)
        df_good_hole = df.loc[(df['DCAL'] <= upper)&(df['DCAL'] >= lower)] #good hole
        return df_good_hole

    #Hiệu chỉnh đường log về Min Max scale dải giá trị. từ 0 đến 1 
    def MinMax_scale (df, curve_name):
        '''Hàm này dùng để hiệu chỉnh log curve về dải giá trị từ 0 đến 1'''
        scaler = MinMaxScaler()
        data_minmax_list = []
        
        for well in df['WELL'].unique():
            data_minmaxscale = df.loc[df['WELL'] == well].copy()
            data_minmaxscale[f'{curve_name}_Raw'] = df[curve_name]
            data_minmaxscale[curve_name] = scaler.fit_transform(data_minmaxscale[[curve_name]])
            data_minmax_list.append(data_minmaxscale)
            Minmax_scaled_df = pd.concat(data_minmax_list, sort=True)
            return Minmax_scaled_df

    #Hiệu chỉnh đường log theo công thức 2 điểm

    #Tính các giá trị quantile của đường log được chọn để hiệu chỉnh cho các giếng khoan 
    def calculated_quantile (df, curve_name, min_quantile_value,max_quantile_value):
        '''Hàm này dùng để tính các giá trị quantile của đường log'''
        min_quantile_curve = df.groupby('WELL')[curve_name].quantile(min_quantile_value)
        max_quantile_curve = df.groupby('WELL')[curve_name].quantile(max_quantile_value)
        return min_quantile_curve, max_quantile_curve

    #Tạo các cột mới well_low, well_high
    # def quantile_new_col(df, well_low, well_high, curve_name, min_quantile_value, max_quantile_value):
    def quantile_new_col(df, curve_name, min_quantile_value, max_quantile_value):
        '''Hàm này dùng để thêm cột đã tính các giá trị quantile vào dataframe chứa dữ liệu đang cần normalize'''
        min_quantile_curve, max_quantile_curve = calculated_quantile(df, curve_name, min_quantile_value, max_quantile_value)
        df["well_low"] = df['WELL'].map(min_quantile_curve)
        df["well_high"] = df['WELL'].map(max_quantile_curve)
        return df["well_low"], df["well_high"]

        #Hàm hiệu chỉnh đường log bằng cách  shift ngang
    def normalize_shifting_curve(df, curve_name, ref_low, well_low, normalized_curve_name):
        def normalize_shifting(a, curve_val, ref_low, well_low):
            return float(a) * curve_val + (ref_low - well_low)
        
        df[normalized_curve_name] = df.apply(lambda x: normalize_shifting(a, x[curve_name], ref_low, x["well_low"]), axis=1)
        return df

    def gmm_analysis(data, curve_name):
        '''Hàm này để sử dụng thuật toán Gaussian Mixture Model (GMM) tính quantile cho 1 đường log của tập hợp các giếng khoan đã được lựa chọn'''
        # Prepare data
        X = data[curve_name].values.reshape(-1, 1)

        # Build and fit GMM model
        n_components = 1
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(X)

        # Get representative distribution parameters
        means = gmm.means_
        covariances = gmm.covariances_

        # Generate synthetic samples from the fitted GMM
        samples, _ = gmm.sample(X.shape[0])

        # Flatten the samples array
        samples = samples.flatten()

        # Compute percentiles
        p5 = np.percentile(samples, 5)
        p95 = np.percentile(samples, 95)
        pmin = np.percentile(samples, float(min_quantile_value)*100)
        pmax = np.percentile(samples, float(max_quantile_value)*100)

        # Print the results
        print("Representative Distribution from Gaussian Mixture Model (GMM):")
        print(f"Mean: {means[0][0]:.2f}")
        print(f"Covariance: {covariances[0][0][0]:.2f}")
        print("Percentiles:")
        print("P5: {:.2f}".format(p5))
        print(f"P{float(min_quantile_value)*100}: {pmin:.2f}")
        print(f"P{float(max_quantile_value)*100}: {pmax:.2f}")
        print("P95: {:.2f}".format(pmax))
        return p5, p95, pmin, pmax

    #Điền các giá trị reference low, reference high của reference well (giếng khoan tiêu biểu được lựa chọn) từ các giá trị quantile đã tính 
    #Hàm áp dụng thuật toán GMM để tính các giá trị quantile của đường log

    def normalize_curve(df, curve_name, ref_low, ref_high, well_low, well_high, normalized_curve_name):
        # Convert input parameters to float
        ref_low = float(ref_low) if ref_low is not None else 0.0
        ref_high = float(ref_high) if ref_high is not None else 0.0
        well_low = float(well_low) if well_low is not None else 0.0
        well_high = float(well_high) if well_high is not None else 0.0

        def normalize(curve_val, ref_low, ref_high, well_low, well_high):
            return ref_low + ((ref_high - ref_low) * ((curve_val - well_low) / (well_high - well_low)))

        df[normalized_curve_name] = df.apply(lambda x: normalize(
            float(x[curve_name]) if x[curve_name] is not None else 0.0,
            ref_low, ref_high, 
            float(x["well_low"]) if x["well_low"] is not None else 0.0, 
            float(x["well_high"]) if x["well_high"] is not None else 0.0), axis=1)
        
        return df

    # # Import the data
    # df = las_to_df(path, sel_wells)
    df = df_full
    # Display the well list
    print(well_list(df))

    # # Show data info
    # data_info(df)

    # Drop NaNs
    # subset_cols = data_info(df)
    df = drop_na_subset(df, curve_name)

    # Filter data by formation and depth
    if filter_data == "Formation":
        print ("Fillter_data by formation")
        # Process marker table and add formations
        df_formation = marker_table(path_df, top, base, well, formation)
        # print (df_formation)
        df['Formation'] = df.apply(add_formations_to_df, axis=1)
        df = filtered_formation(df, formation_column, formation_name)
    elif filter_data == "Depth":
        df = filtered_depth(df, depth_column, float(min_depth), float(max_depth))
    elif filter_data == "Both":
        df_formation = marker_table(path_df, top, base, well, formation)
        # print (df_formation)
        df['Formation'] = df.apply(add_formations_to_df, axis=1)
        df = filtered_formation(df, formation_column, formation_name)
        df = filtered_depth(df, depth_column, float(min_depth), float(max_depth))
    else:
        pass

    # Đánh giá range của các giếng và loại bỏ những giếng out of range
    excluded_wells = calculate_percentage_outside_range(df, curve_name, float(threshold_outside_range))
    filtered_data= df[~df['WELL'].isin(excluded_wells['WELL'].unique())]
    print("Các giếng bị loại ra:", filtered_data['WELL'].unique())
    print("Các giếng còn lại:", excluded_wells['WELL'].unique())

    # Remove certain wells
    wells_to_remove = filtered_data['WELL'].unique()
    df = remove_well(df, wells_to_remove)

    if zscore == "On":
    # Process zscore data and remove outliers
        # zscore_threshold = 90
        df = process_zscore_data(df, curve_name, zscore_threshold)
        df = remove_outliers(df)
    else:
        pass
 
    if upper_value_DCAL is not None and lower_value_DCAL is not None:
        # Filter data by good hole conditions
        df = on_gauge_hole(df, upper_value_DCAL, lower_value_DCAL)
    else:
        df = on_gauge_hole(df, 1, -1)

    # Apply min-max scaling
    if MinMax_scale =='On':
        print ("Application 0-1 scale for Normalization")
        df = MinMax_scale(df, curve_name)
    else:
        print ("Application default scale for Normalization")
    # Calculate quantiles and normalize data #Stack
    well_low = None
    well_high = None

    df["well_low"], df["well_high"] = quantile_new_col(df, curve_name, min_quantile_value, max_quantile_value)

    if algorithm =='On':
        print("Apply reference wells by Algorithm")
        # Calculate quantiles and normalize data based on algorithm
        _,_,ref_low,ref_high = gmm_analysis(df, curve_name)
    else: 
        print("Apply reference wells by User")
        df = normalize_curve(df, curve_name, int(ref_low), int(ref_high), well_low, well_high, normalized_curve_name)
    # # Apply shifting normalization
    if linear_shift == "On":
        print("Apply a point for Normalization")
        #Hàm hiệu chỉnh đường log bằng cách  shift ngang
        _,_,ref_low, ref_high = gmm_analysis(df, curve_name)
        df = normalize_shifting_curve(df, curve_name, ref_low, well_low, normalized_curve_name)
    else: 
        print("Apply two points for Normalization")
        df = normalize_curve(df, curve_name, ref_low, ref_high, well_low, well_high, normalized_curve_name)

    df_nor = df[[normalized_curve_name, curve_name, "WELL"]]
    print ('Danh sách giếng khoan đã được chuẩn hóa:', df_nor['WELL'].unique())
    print ("Done processing data, moving to return output result")

    # API return for output
    custom_result = {}
    # if parameter.get("ref_low"):
    #     custom_result["ref_low"] = ref_low

    # if parameter.get("ref_high"):
    #     custom_result["ref_high"] = ref_high

  # if parameter.get("df"):
    #     custom_result["df"] = df
    if parameter.get("removed_well"):
        custom_result["removed_well"] = filtered_data['WELL'].unique()

    if parameter.get("remain_well"):
        custom_result["remain_well"] = excluded_wells['WELL'].unique()

    if parameter.get("df_nor"):
        custom_result["df_nor"] = df_nor
    result_in_json_full = {
        **parameter,
        **custom_result
    }
    import json
    def json_serializable(val):
        if isinstance(val, pd.Series):
            return val.to_dict()
        elif isinstance(val, pd.DataFrame):
            return val.to_dict(orient='split')
        elif isinstance(val, np.ndarray):
            return val.tolist()
        else:
            return val
    result_in_json_full = {k: json_serializable(v) for k, v in result_in_json_full.items()}
    result_in_json = json.dumps(result_in_json_full)
    # Lưu cấu trúc dữ liệu vào tệp tin JSON
    # with open('/lakehouse/default/Files/result_in_json.json', 'w') as file:
    #     json.dump(result_in_json, file)

    # Phân tích chuỗi JSON thành một dictionary Python
    python_dict = json.loads(result_in_json)

    # Đếm tổng số cả khóa và giá trị trong dictionary
    total = 0
    for key, value in python_dict.items():
        total += 1  # Tăng tổng lên 1 cho mỗi khóa
        total += 1  # Tăng tổng lên 1 cho mỗi giá trị

    print(f"Tổng số cả khóa và giá trị trong chuỗi JSON: {total}")
    # result_in_json_full = pd.read_csv(df_full)
    result_in_json = json.dumps(result_in_json_full)
    return result_in_json
def normalize_curve(df_full, parameter):
    import sys
    sys.path.append('/lakehouse/default/Files')
    sys.path.append('../')
    #Nhập các thư viện cần thiết, để xây dựng thuật toán
    import pandas as pd
    import numpy as np

    from sklearn.mixture import GaussianMixture

#     #Setup parameter for API
    algorithm = parameter.get("algorithm") #algorithm #"Auto selection' #'Manual'
    linear_shift = parameter.get("linear_shift")

    #Chọn lọc và lựa chọn dữ liệu cho quá trình chuẩn hoá 
    min_quantile_value = parameter.get("min_quantile_value") #Người dùng nhập liệu giá trị xác xuất nhỏ nhất muốn xác định (theo tính năng người dùng tự nhập liệu)
    max_quantile_value = parameter.get("max_quantile_value") #Người dùng nhập liệu giá trị xác xuất lớn nhất muốn xác định (theo tính năng người dùng tự nhập liệu)
    ref_low = parameter.get("ref_low") #Người dùng nhập liệu giá trị xác xuất lớn nhất muốn xác định (theo tính năng người dùng tự nhập liệu)
    ref_high = parameter.get("ref_high") #Người dùng nhập liệu giá trị xác xuất nhỏ nhất muốn xác định (theo tính năng người dùng tự nhập liệu)
    a = parameter.get("a") #Tham số áp dụng cho quá trình chuẩn hoá một điểm

    #Người dùng xác định đường cong sử dụng để chuẩn hoá
    # column_name = parameter.get("column_name") #Tên cột đường cong muốn chuẩn hoá
    curve_name =  parameter.get("curve_name") #Tên đường cong muốn chuẩn hoá
    normalized_curve_name = parameter.get("normalized_curve_name") #Tên đường cong sau khi chuẩn hoá

    #Danh sách giếng khoan được nhập vào

    #Loại bỏ các điểm null data trong đường log
    def drop_na_subset(df, curve_name):
        """
        Hàm này loại bỏ các hàng có giá trị NaN trong các cột con trong DataFrame df.
        Arguments:
        df -- DataFrame đầu vào
        subset_cols -- Danh sách tên các cột con để kiểm tra giá trị NaN  
        Returns:
        DataFrame -- DataFrame mới đã loại bỏ các hàng có giá trị NaN
        """
        return df.dropna(subset=curve_name)
    
    #Hiệu chỉnh đường log theo công thức 2 điểm

    #Tính các giá trị quantile của đường log được chọn để hiệu chỉnh cho các giếng khoan 
    def calculated_quantile (df, curve_name, min_quantile_value,max_quantile_value):
        '''Hàm này dùng để tính các giá trị quantile của đường log'''
        min_quantile_curve = df.groupby('WELL')[curve_name].quantile(min_quantile_value)
        max_quantile_curve = df.groupby('WELL')[curve_name].quantile(max_quantile_value)
        return min_quantile_curve, max_quantile_curve

    #Tạo các cột mới well_low, well_high
    # def quantile_new_col(df, well_low, well_high, curve_name, min_quantile_value, max_quantile_value):
    def quantile_new_col(df, curve_name, min_quantile_value, max_quantile_value):
        '''Hàm này dùng để thêm cột đã tính các giá trị quantile vào dataframe chứa dữ liệu đang cần normalize'''
        min_quantile_curve, max_quantile_curve = calculated_quantile(df, curve_name, min_quantile_value, max_quantile_value)
        df["well_low"] = df['WELL'].map(min_quantile_curve)
        df["well_high"] = df['WELL'].map(max_quantile_curve)
        return df["well_low"], df["well_high"]

        #Hàm hiệu chỉnh đường log bằng cách  shift ngang
    def normalize_shifting_curve(df, curve_name, ref_low, well_low, normalized_curve_name):
        def normalize_shifting(a, curve_val, ref_low, well_low):
            return float(a) * curve_val + (ref_low - well_low)
        
        df[normalized_curve_name] = df.apply(lambda x: normalize_shifting(a, x[curve_name], ref_low, x["well_low"]), axis=1)
        return df

    def gmm_analysis(data, curve_name):
        '''Hàm này để sử dụng thuật toán Gaussian Mixture Model (GMM) tính quantile cho 1 đường log của tập hợp các giếng khoan đã được lựa chọn'''
        # Prepare data
        X = data[curve_name].values.reshape(-1, 1)

        # Build and fit GMM model
        n_components = 1
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(X)

        # Get representative distribution parameters
        means = gmm.means_
        covariances = gmm.covariances_

        # Generate synthetic samples from the fitted GMM
        samples, _ = gmm.sample(X.shape[0])

        # Flatten the samples array
        samples = samples.flatten()

        # Compute percentiles
        p5 = np.percentile(samples, 5)
        p95 = np.percentile(samples, 95)
        pmin = np.percentile(samples, float(min_quantile_value)*100)
        pmax = np.percentile(samples, float(max_quantile_value)*100)

        # Print the results
        print("Representative Distribution from Gaussian Mixture Model (GMM):")
        print(f"Mean: {means[0][0]:.2f}")
        print(f"Covariance: {covariances[0][0][0]:.2f}")
        print("Percentiles:")
        print("P5: {:.2f}".format(p5))
        print(f"P{float(min_quantile_value)*100}: {pmin:.2f}")
        print(f"P{float(max_quantile_value)*100}: {pmax:.2f}")
        print("P95: {:.2f}".format(pmax))
        return p5, p95, pmin, pmax

    #Điền các giá trị reference low, reference high của reference well (giếng khoan tiêu biểu được lựa chọn) từ các giá trị quantile đã tính 
    #Hàm áp dụng thuật toán GMM để tính các giá trị quantile của đường log

    def normalize_curve(df, curve_name, ref_low, ref_high, well_low, well_high, normalized_curve_name):
        # Convert input parameters to float
        ref_low = float(ref_low) if ref_low is not None else 0.0
        ref_high = float(ref_high) if ref_high is not None else 0.0
        well_low = float(well_low) if well_low is not None else 0.0
        well_high = float(well_high) if well_high is not None else 0.0

        def normalize(curve_val, ref_low, ref_high, well_low, well_high):
            return ref_low + ((ref_high - ref_low) * ((curve_val - well_low) / (well_high - well_low)))

        df[normalized_curve_name] = df.apply(lambda x: normalize(
            float(x[curve_name]) if x[curve_name] is not None else 0.0,
            ref_low, ref_high, 
            float(x["well_low"]) if x["well_low"] is not None else 0.0, 
            float(x["well_high"]) if x["well_high"] is not None else 0.0), axis=1)
        
        return df

    df = df_full

    df = drop_na_subset(df, curve_name)

    # Calculate quantiles and normalize data #Stack
    well_low = None
    well_high = None

    df["well_low"], df["well_high"] = quantile_new_col(df, curve_name, min_quantile_value, max_quantile_value)

    if algorithm =='On':
        print("Apply reference wells by Algorithm")
        # Calculate quantiles and normalize data based on algorithm
        _,_,ref_low,ref_high = gmm_analysis(df, curve_name)
    else: 
        print("Apply reference wells by User")
        df = normalize_curve(df, curve_name, int(ref_low), int(ref_high), well_low, well_high, normalized_curve_name)
    # # Apply shifting normalization
    if linear_shift == "On":
        print("Apply a point for Normalization")
        #Hàm hiệu chỉnh đường log bằng cách  shift ngang
        _,_,ref_low, ref_high = gmm_analysis(df, curve_name)
        df = normalize_shifting_curve(df, curve_name, ref_low, well_low, normalized_curve_name)
    else: 
        print("Apply two points for Normalization")
        df = normalize_curve(df, curve_name, ref_low, ref_high, well_low, well_high, normalized_curve_name)

    df_nor = df[[normalized_curve_name, curve_name, "WELL"]]
    print ('Danh sách giếng khoan đã được chuẩn hóa:', df_nor['WELL'].unique())
    print ("Done processing data, moving to return output result")

    # API return for output
    custom_result = {}

    if parameter.get("df_nor"):
        custom_result["df_nor"] = df_nor
    result_in_json_full = {
        **parameter,
        **custom_result
    }
    import json
    def json_serializable(val):
        if isinstance(val, pd.Series):
            return val.to_dict()
        elif isinstance(val, pd.DataFrame):
            return val.to_dict(orient='split')
        elif isinstance(val, np.ndarray):
            return val.tolist()
        else:
            return val
    result_in_json_full = {k: json_serializable(v) for k, v in result_in_json_full.items()}

    # Lưu cấu trúc dữ liệu vào tệp tin JSON
    result_in_json = json.dumps(result_in_json_full)
    return result_in_json
