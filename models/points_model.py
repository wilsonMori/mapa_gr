from sklearn.cluster import KMeans

class PointsModel:
    def __init__(self, df):
        self.df = df

    def assign_to_technicians(self, df, tecnicos):
        kmeans = KMeans(n_clusters=tecnicos, random_state=42).fit(df[['Latitud','Longitud']])
        df['Tecnico'] = kmeans.labels_
        return df
