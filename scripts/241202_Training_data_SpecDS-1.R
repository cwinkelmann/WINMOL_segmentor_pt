
library(raster)
library(geosphere)
library(rgdal)
library(magick)
library(tidyverse)
library(sp)
library(foreach)
library(doParallel)
library(spatialEco)
library(sf)
library(rgeos)
#install.packages("rgeos")
#install.packages("spatialEco")

rgdal::setCPLConfigOption("GDAL_PAM_ENABLED", "FALSE")
##### Set up environment #####
gc()
responseCol <- "stem_id"
cordinates <- 32632


  ##############################################################################  
  ###########################Create SpecDS and TestDS###################################### 
  ##############################################################################
  
  
  imgpath="P:/WINMOL_Train_Gen/Trainigsdatensatz_v3/Quesenbank_xxx/ortho_clip_quesenbank_20221013.tif"
  training_data="P:/WINMOL_Train_Gen/Trainigsdatensatz_v3/Quesenbank_xxx/trainings_stems_quesenbank_xxx.shp"
  windthrow_area="P:/WINMOL_Train_Gen/Trainigsdatensatz_v3/Quesenbank_xxx/trainingscluster_quesenbank_xxx.shp"
  outdir <-"P:/WINMOL_Train_Gen/output/SpecDS_UNet_512_20241219_5/"
  #Load orthophoto
  img <- brick(imgpath)
  crs(img)<-cordinates
  #Load shape file with training data
  
  maskdir <- paste(outdir,"mask/",sep="")
  traindir <- paste(outdir,"train/",sep="")

  SpecData <- shapefile(training_data)
  crs(SpecData)<-cordinates
 
  STORM_AREA <- shapefile(windthrow_area)
  SpecData@bbox<-as.matrix(extent(STORM_AREA))
  STORM_AREA <-st_buffer(st_as_sf(STORM_AREA),-11)
  STORM_AREA <- as(STORM_AREA, 'Spatial')
  crs(STORM_AREA)<-cordinates
  
  SmplExtend <- 15
  SmplGeom <- 512
  smplNbr <- floor((area(STORM_AREA)/(SmplExtend^2)*100))
  
  start_time <- Sys.time()
  
  ncores <- detectCores()
  cl <- makePSOCKcluster(ncores-16) 
  
  clusterEvalQ(cl, {
    
    library(raster)
  #  library(geosphere)
  #  library(rgdal)
    library(magick)
  #  library(tidyverse)
  #  library(sp)
    library(spatialEco)
  #  library(rgeos)
    library(sf)
  })
  registerDoParallel(cl) 
  
  
#  foreach(i=1:smplNbr, .export="img") %dopar%{
  foreach(i=1:smplNbr) %dopar%{
    time_stemp=paste(Sys.getpid(),"_",Sys.info()[['nodename']],format(Sys.time(), '%H%M%OS6'), sep="")
    
    rdmCoord=spsample(STORM_AREA,1,type = 'random')
    rdmAngle=runif(n=1, min = -179, max = 180)
    
    sample_footprint=as(extent(rdmCoord@coords[1,1]-SmplExtend/2,rdmCoord@coords[1,1]+SmplExtend/2,rdmCoord@coords[1,2]-SmplExtend/2,rdmCoord@coords[1,2]+SmplExtend/2), 'SpatialPolygons')
    sample_footprint=rotate.polygon(sample_footprint, angle=rdmAngle, sp=TRUE)
    sample_footprint@proj4string<-CRS(as.character("+init=epsg:32632"))
    train_img <- crop(img, sample_footprint)
    train_img <- mask(train_img, sample_footprint)
  #  mask_shp<-crop(SpecData, sample_footprint)
    #mask_shp<-intersect(SpecData, sample_footprint)
    mask_shp_sf=spatial.select(SpecData, sample_footprint)
    if(length(mask_shp_sf$id)>0){
      mask_shp=as(mask_shp_sf, "Spatial")

    if(sum(area(mask_shp))>SmplExtend^2/200){ 
    #  s <- raster(extent(train_img),nrow=1024,ncol=1024)
  #    crs(s)<-train_img@crs
   #   train_img <- raster::resample(train_img,s, method='bilinear')
      train_img_<-as.array(train_img)
      train_img_<-magick::image_read(train_img_ /255) #/65535)# %>% magick::image_scale(paste("!",SmplGeom,"x",SmplGeom, sep=""))
      
      train_img_ <- image_transparent(train_img_, color = "black")  
      train_img_ <- image_background(train_img_, color = "transparent") 
      train_img_ = image_trim(train_img_)
      train_img_ = image_rotate(train_img_,-1*rdmAngle)
      train_img_ <- image_transparent(train_img_, color = "black")  
      train_img_ <- image_background(train_img_, color = "transparent") 
      train_img_ = image_trim(train_img_)
      train_img_ = magick::image_scale(train_img_,paste("!",SmplGeom,"x",SmplGeom, sep=""))
      
      mask <- rasterize(mask_shp, train_img,field=255,background=75)
      #mask_shp<-intersect(mask_shp,sample_footprint)
      #rotate.polygon(mask_shp, angle=-1*rdmAngle, sp=TRUE)
      mask <- crop(mask, sample_footprint)
      mask <- mask(mask, sample_footprint)
        
      #convert to img
      
      mask<-as.array(mask)
      mask<-magick::image_read(mask / 255)# %>% magick::image_scale(paste("!",SmplGeom,"x",SmplGeom, sep=""))
      mask <- image_transparent(mask, color = "black")
      mask <- image_background(mask, color = "transparent")
      mask = image_trim(mask)
      mask = image_rotate(mask,-1*rdmAngle)
      mask <- image_transparent(mask, color = "black")
      mask <- image_background(mask, color = "transparent")
      mask = image_trim(mask)
        
      mask=magick::image_scale(mask,paste("!",SmplGeom,"x",SmplGeom, sep=""))
      mask=image_ordered_dither(mask,threshold_map= "2x1")
      mask <- image_background(mask, color = "black")
        #export
      image_write(train_img_, path = paste(traindir,"train_",i,"_",time_stemp,".jpeg",sep=""), format = "jpeg")
      image_write(mask, path =paste(maskdir,"mask_",i,"_",time_stemp,".gif",sep=""), format = "gif")
      }

      
  }
        
  }

  stopCluster(cl)
  ##
  end_time <- Sys.time()
  SpecDS_time<-end_time - start_time
  print(SpecDS_time)

  
 