def array_2_plot(array):
    '''
    각 shapefile을 변환된 DEM 영역에 맞춰 변환한 array가 input
    '''
    array = pd.DataFrame(array)
    fig, ax = plt.subplots(figsize=(20, 20)) 
    ax.imshow(watershed_basins_transformed, cmap='gray', interpolation='none') 
    ax.set_title('Array Visualization')
    plt.show()