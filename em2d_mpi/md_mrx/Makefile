LIB_DIR = ..
include $(LIB_DIR)/Makefile_inc
TARGET = a.out
OBJS = const.o init.o main.o
.PHONY : all 
.PHONY : clean
.SUFFIXES :
.SUFFIXES : .o .f90
.f90.o:
	$(FC) -L$(LIB_DIR) $(FFLAGS) -I $(LIB_DIR)/common -c $< -lem2d

all: $(TARGET)

$(TARGET) : $(OBJS)
	$(FC) -L$(LIB_DIR) -o $(TARGET) $(FFLAGS) $(OBJS) -lem2d 

$(LIB_DIR)/libem2d.a : 
	cd $(LIB_DIR) ; make

main.o : init.o const.o $(LIB_DIR)/libem2d.a
init.o : const.o $(LIB_DIR)/libem2d.a

############# PSD calculation ###############
# ここで開始ステップを指定できます
MIN_STEP = 26000

psd_calc : $(LIB_DIR)/psd/psd.out
	$(LIB_DIR)/psd/psd.out 160 64 ./ $$( \
		ls /data/shok/dat/*_rank*.dat | \
		awk -F/ '{split($$NF, a, "_"); if (a[1] + 0 >= $(MIN_STEP)) print $$0}' \
	)

$(LIB_DIR)/psd/psd.out :
	cd $(LIB_DIR)/psd ; make
###############################################

############# PSD calculation ###############
psd_calc : $(LIB_DIR)/psd/psd.out
	$(LIB_DIR)/psd/psd.out 160 64 ./ /data/shok/dat/*_rank*.dat

$(LIB_DIR)/psd/psd.out :
	cd $(LIB_DIR)/psd ; make
###############################################

clean :
	rm -f $(TARGET) $(OBJS) *.mod dat/*.dat mom/*.dat psd/*.dat

