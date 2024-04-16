#include "cholesky.h"





//dA:输入被分解方阵
//dC:cholesky分解结果方阵
//trans -> false: col major; true: row major
//uplo -> false: lower; true: upper
//ldda：leading dimension
//batch=1
mluOpStatus_t MLUOP_WIN_API 
mluOpCholesky(mluOpHandle_t handle,const mluOpTensorDescriptor_t input_desc,float* d_input, const mluOpTensorDescriptor_t output_desc, float* d_output,bool upper)
{
    PARAM_CHECK("mluOpCholesky", handle != NULL);
    PARAM_CHECK("mluOpCholesky", input_desc != NULL);
    PARAM_CHECK("mluOpCholesky", d_input != NULL);
    PARAM_CHECK("mluOpCholesky", output_desc != NULL);
    PARAM_CHECK("mluOpCholesky", d_output != NULL);

    PARAM_CHECK("mluOpCholesky", input_desc->dim == 2||input_desc->dim == 3);
    PARAM_CHECK("mluOpCholesky", output_desc->dim == input_desc->dim);
    PARAM_CHECK("mluOpCholesky", input_desc->dims[0] > 0);
    PARAM_CHECK("mluOpCholesky", input_desc->dims[1] > 0);
    PARAM_CHECK("mluOpCholesky", output_desc->dims[0] > 0);
    PARAM_CHECK("mluOpCholesky", output_desc->dims[1] > 0);
    if(input_desc->dim == 3)
    {
        PARAM_CHECK("mluOpCholesky", input_desc->dims[2] > 0);
        PARAM_CHECK("mluOpCholesky", output_desc->dims[2] > 0);
    }
    
    
    int recnb = REC_NB;
    int gbstep = 0;
    int dim = input_desc->dim;
    bool is_row_major = (input_desc->strides)[dim-1]==1;
    
    int size_a = 0, lda = 0, size_c = 0, ldc = 0;
    int batch_size = 1;
    if(dim == 2)
    {
        size_a = input_desc->dims[0];
        lda = input_desc->dims[1];
        size_c = output_desc->dims[0];
        ldc = output_desc->dims[1];
    }
    else if(dim == 3)
    {
        batch_size = input_desc->dims[0];
        size_a = input_desc->dims[1];
        lda = input_desc->dims[2];
        size_c = output_desc->dims[1];
        ldc = output_desc->dims[2];
    }
    
    
    float* work_space;
    float* work_space_h;
    CNRT_CHECK(cnrtMalloc((void **)&work_space, NB*NB*sizeof(float)));
    CNRT_CHECK(cnrtMemset(work_space, 0, NB*NB*sizeof(float)));
    work_space_h = (float*)malloc(NB*NB*sizeof(float));
    PARAM_CHECK("mluOpCholesky", lda >= size_a);
    PARAM_CHECK("mluOpCholesky", ldc >= size_c);

    cnrtQueue_t queue;
    mluOpGetQueue(handle,&queue);
    // CNRT_CHECK(cnrtSetDevice(0));
    // CNRT_CHECK(cnrtQueueCreate(&queue));

    // cnrtNotifier_t start, end;
    // CNRT_CHECK(cnrtNotifierCreate(&start));
    // CNRT_CHECK(cnrtNotifierCreate(&end));

    int jb;
    const float s_one = 1.0;
    const float s_neg_one = -1.0;
        
    if(upper == true)
    {
        printf("start transpose 1\n");
        CHECK_RETURN("mluOpCholesky",
                transpose(size_a,d_input,d_output,handle));
    }
    else 
    {
        CNRT_CHECK(cnrtMemcpy(d_output, d_input, sizeof(float)*size_a*lda, CNRT_MEM_TRANS_DIR_DEV2DEV));
    }
    cnrtQueueSync(queue);

    //TODO:检查拷贝开销
    
    // if(upper == true)
    // {
    //     CHECK_RETURN("mluOpCholesky",
    //             transpose(size_a,d_input,d_output,handle));
    //     //print d_output
    //     cnrtMemcpy(work_space_h, d_output, sizeof(float)*size_a*size_a, CNRT_MEM_TRANS_DIR_DEV2HOST);
    //     //print work_space_h
    //     // printf("matrix after transpose:\n");
    //     // for(int i = 0; i < size_a; i++)
    //     // {
    //     //     for(int j = 0; j < size_a; j++)
    //     //     {
    //     //         printf("%.2f ",work_space_h[i*size_a+j]);
    //     //     }
    //     //     printf("\n");
    //     // }
        
    // }
    // else
    // {
        int row = is_row_major ? lda : size_a;
        // int nb = row > 512 ? NB : (NB/2);
        int nb = NB;
        for(int j = 0; j < row; j+=nb)
        {
            jb = std::min(nb, row-j);
            CHECK_RETURN("mluOpCholesky",
                ssyrk(false,is_row_major,jb,j,OFFSET_ROW(d_output,j,0),lda,OFFSET_ROW(d_output,j,j),lda,handle));
            cnrtQueueSync(queue);
            CHECK_RETURN("mluOpCholesky",
                mlu_spotrf_rectile(is_row_major,false,jb,recnb,OFFSET_ROW(d_output,j,j),lda,j, handle));
            // cnrtQueueSync(queue);
            if(j+jb < row)
            {
                CHECK_RETURN("mluOpCholesky",
                    sgemm(!is_row_major,is_row_major,row-j-jb,jb,j,-1.0f,1.0f,
                        OFFSET_ROW(d_output,j+jb,0),lda,
                        OFFSET_ROW(d_output,j,0),lda,
                        OFFSET_ROW(d_output,j+jb,j),lda, handle));
                cnrtQueueSync(queue);
            }    
            if(j+jb < row)
            {
                CHECK_RETURN("mluOpCholesky",
                    strsm(false,is_row_major,jb,row-j-jb,OFFSET_ROW(d_output,j,j),lda,OFFSET_ROW(d_output,j+jb,j),lda, handle));
                cnrtQueueSync(queue);
            }
        }
    // }
    
    if(upper)
    {
        cnrtQueueSync(queue);
        CHECK_RETURN("mluOpCholesky",
                transpose(size_a,d_output,d_output,handle));
    }
    

    

    cnrtQueueSync(queue);

    // cnrtMemcpy(work_space_h, work_space, sizeof(float)*NB*NB, CNRT_MEM_TRANS_DIR_DEV2HOST);
    //print work_space_h
    // printf("work_space:\n");
    // for(int i = 0; i < NB; i++)
    // {
    //     for(int j = 0; j < NB; j++)
    //     {
    //         printf("%.2f ",work_space_h[i*NB+j]);
    //     }
    //     printf("\n");
    // }

    return MLUOP_STATUS_SUCCESS;
}