import datetime
import s3fs
def get_patmos_s3_files(year, month, day, satellite, ascdes):
    """
    Process data and return a list.

    Args:
        year (int):  Year
        month (int): Month of year: range = [1,12]
        day (int):   Day of month range: [1-31]
        satellite (str): NOAA satellite name. E.g., 'NOAA-14'
        ascdec (str):    Should either be 'asc' or 'des'

    Returns:
        URLs: Return the s3 list of URLs 
    """
    #convert date to object
    date_now = datetime.date(year=year, month=month, day=day)
    date_bef = date_now - datetime.timedelta(days=1)
    date_aft = date_now + datetime.timedelta(days=1)

    str_date_now = date_now.strftime("%Y%m%d")
    str_date_bef = date_bef.strftime("%Y%m%d")
    str_date_aft = date_aft.strftime("%Y%m%d")

    yearbef = date_bef.strftime("%Y")
    yearnow = date_now.strftime("%Y")
    yearaft = date_aft.strftime("%Y")
    
    # initialize patmos s3 bucket and run globs
    s3_globbef='s3://noaa-cdr-patmosx-radiances-and-clouds-pds/data/'+yearbef+'/patmosx_v06r00_'+satellite+'_'+ascdes+'_*'+str_date_bef+'*nc'
    s3_globnow='s3://noaa-cdr-patmosx-radiances-and-clouds-pds/data/'+yearnow+'/patmosx_v06r00_'+satellite+'_'+ascdes+'_*'+str_date_now+'*nc'
    s3_globaft='s3://noaa-cdr-patmosx-radiances-and-clouds-pds/data/'+yearaft+'/patmosx_v06r00_'+satellite+'_'+ascdes+'_*'+str_date_aft+'*nc'

    s3 = s3fs.S3FileSystem(anon=True)

    urls_bef = ['s3://' + f for f in s3.glob(s3_globbef)]
    urls_now = ['s3://' + f for f in s3.glob(s3_globnow)]
    urls_aft = ['s3://' + f for f in s3.glob(s3_globaft)]
    #print()
    #print(urls_bef)
    #print(urls_now)
    #print(urls_aft)
    #print()
    if not urls_now:
        print("No files found matching the glob expression:\n",s3_globnow)
        #Check which satellites are available for this date:
        s3_avail='s3://noaa-cdr-patmosx-radiances-and-clouds-pds/data/'+yearnow+'/patmosx_v06r00_*_asc_*'+str_date_now+'*nc'
        files=s3.glob(s3_avail)
        print("The following satellites are available:")
        for f in files:
            parts=f.split("_")
            print("Satellite: "+parts[2])
    else:
        print("url yesterday: " + urls_bef[0])
        print("url today:     " + urls_now[0])
        print("url tomorrow:  " + urls_aft[0])
    
    
    # Your processing logic goes here
#    processed_data = [int1, int2, int3, str1, str2]
    URLs = [urls_bef[0], urls_now[0], urls_aft[0]]
    return URLs

