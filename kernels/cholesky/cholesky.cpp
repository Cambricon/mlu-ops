#include "cholesky.h"

//dA:输入被分解方阵
//dC:cholesky分解结果方阵
//trans -> false: col major; true: row major
//uplo -> false: lower; true: upper
//ldda：leading dimension

mluOpStatus_t MLUOP_WIN_API mluOpGetCholeskyWorkspace(mluOpTensorDescriptor_t input_desc, size_t* size, float** workspace)
{
    PARAM_CHECK("mluOpCholesky", input_desc != NULL);


    PARAM_CHECK("mluOpCholesky", input_desc->dim == 2||input_desc->dim == 3);
    PARAM_CHECK("mluOpCholesky", input_desc->dims[0] > 0);
    PARAM_CHECK("mluOpCholesky", input_desc->dims[1] > 0);

    if(input_desc->dim == 3)
    {
        PARAM_CHECK("mluOpCholesky", input_desc->dims[2] > 0);
    }

    mluOpDataType_t dtype = input_desc->dtype;
    PARAM_CHECK("mluOpCholesky", dtype == MLUOP_DTYPE_FLOAT || dtype == MLUOP_DTYPE_COMPLEX_FLOAT);

    int type_size = (dtype == MLUOP_DTYPE_FLOAT) ? 4 : 8;
    int size_a = 0, lda = 0, size_c = 0, ldc = 0;
    int batch_size = 1;
    int dim = input_desc->dim;
    if(dim == 2)
    {
        size_a = input_desc->dims[0];
    }
    else if(dim == 3)
    {
        batch_size = input_desc->dims[0];
        size_a = input_desc->dims[1];
    }
    printf("fuck you!");
    
    if (dtype == MLUOP_DTYPE_FLOAT)
    {
        // *size = size_a*size_a*sizeof(float);
        *size = 0;
    }
    else
    {
        *size = size_a*size_a*sizeof(float)*2*batch_size;
        printf("size:%ul\n",(int)(*size));
    }
    if(*size>0)
    {
        CHECK_RETURN("mluOpCholesky",
                complex_malloc(*size, workspace));
    }
    return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API 
mluOpCholesky(mluOpHandle_t handle,const mluOpTensorDescriptor_t input_desc,float* d_input, const mluOpTensorDescriptor_t output_desc, float* d_output,bool upper, float* workspace)
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

    mluOpDataType_t dtype = input_desc->dtype;
    PARAM_CHECK("mluOpCholesky", dtype == output_desc->dtype);
    PARAM_CHECK("mluOpCholesky", dtype == MLUOP_DTYPE_FLOAT || dtype == MLUOP_DTYPE_COMPLEX_FLOAT);
    
    
    int recnb = REC_NB;
    int gbstep = 0;
    int dim = input_desc->dim;
    bool is_row_major = (input_desc->strides)[dim-1]==1;
    
    
    int type_size = (dtype == MLUOP_DTYPE_FLOAT) ? 4 : 8;
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
    
    if(dtype == MLUOP_DTYPE_FLOAT)
    {
        if(upper == true)
        {
            CHECK_RETURN("mluOpCholesky",
                    transpose(batch_size,size_a,size_a,d_input,d_output,handle));
        }
        else 
        {
            CNRT_CHECK(cnrtMemcpy(d_output, d_input, type_size*size_a*lda*batch_size, CNRT_MEM_TRANS_DIR_DEV2DEV));
        }
    }
    else
    {
        
        CHECK_RETURN("mluOpCholesky",
                    transpose(batch_size,size_a*size_a,2,d_input,d_output,handle));
    }
    
    cnrtQueueSync(queue);

    //TODO:检查拷贝开销
    
    int stride = size_a*lda;
    //printf original matrix
    if(dtype == MLUOP_DTYPE_FLOAT)
    {
        
        int row = is_row_major ? lda : size_a;
        int nb = NB;
        for(int j = 0; j < row; j+=nb)
        {
            jb = std::min(nb, row-j);
            CHECK_RETURN("mluOpCholesky",
                ssyrk(batch_size,stride,false,is_row_major,jb,j,OFFSET_ROW(d_output,j,0),lda,OFFSET_ROW(d_output,j,j),lda,handle));
            cnrtQueueSync(queue);
            CHECK_RETURN("mluOpCholesky",
                mlu_spotrf_rectile(batch_size,stride,is_row_major,false,jb,recnb,OFFSET_ROW(d_output,j,j),lda,j, handle));
            // cnrtQueueSync(queue);
            if(j+jb < row)
            {
                CHECK_RETURN("mluOpCholesky",
                    sgemm(batch_size, !is_row_major,is_row_major,row-j-jb,jb,j,-1.0f,1.0f,
                        OFFSET_ROW(d_output,j+jb,0),lda,stride,
                        OFFSET_ROW(d_output,j,0),lda,stride,
                        OFFSET_ROW(d_output,j+jb,j),lda,stride, handle));
                cnrtQueueSync(queue);
            }    
            if(j+jb < row)
            {
                CHECK_RETURN("mluOpCholesky",
                    strsm(batch_size, stride,false,is_row_major,jb,row-j-jb,OFFSET_ROW(d_output,j,j),lda,OFFSET_ROW(d_output,j+jb,j),lda, handle));
                cnrtQueueSync(queue);
            }
        }
    
        if(upper)
        {
            cnrtQueueSync(queue);
            CHECK_RETURN("mluOpCholesky",
                    transpose(batch_size, size_a,size_a,d_output,d_output,handle));
        }
    }
    else
    {
        recnb = CREC_NB;
        // int nb = NB;
        int nb = NB;
        int row = lda;
        float* r_start = d_output; //实数首地址
        float* i_start = d_output + size_a*lda;//虚数首地址

        set_half_zero(batch_size, size_a*lda, r_start, lda, lda, handle);
        set_half_zero(batch_size, size_a*lda, i_start, lda, lda, handle);

        for(int j = 0; j < row; j+=nb)
        {
            jb = std::min(nb, row-j);
            CHECK_RETURN("mluOpCholesky",
                cherk(batch_size,stride,jb,j,r_start+j*lda,i_start+j*lda,lda,r_start+j*lda+j,i_start+j*lda+j,lda,handle));
            cnrtQueueSync(queue);
            CHECK_RETURN("mluOpCholesky",
                mlu_cpotrf_rectile(batch_size,stride,jb,recnb,r_start+j*lda+j,i_start+j*lda+j,lda, handle));
            // cnrtQueueSync(queue);
            if(j+jb < row)
            {
                CHECK_RETURN("mluOpCholesky",
                    cgemm(batch_size, false,true,row-j-jb,jb,j,-1.0f,1.0f,
                        OFFSET_ROW(r_start,j+jb,0),OFFSET_ROW(i_start,j+jb,0), lda,stride,
                        OFFSET_ROW(r_start,j,0),OFFSET_ROW(i_start,j,0), lda,stride,
                        OFFSET_ROW(r_start,j+jb,j),OFFSET_ROW(i_start,j+jb,j), lda, stride, handle));

                cnrtQueueSync(queue);
            }    
            if(j+jb < row)
            {
                CHECK_RETURN("mluOpCholesky",
                    ctrsm(batch_size, stride,jb,row-j-jb,OFFSET_ROW(r_start,j,j),OFFSET_ROW(i_start,j,j),lda,
                        OFFSET_ROW(r_start,j+jb,j),OFFSET_ROW(i_start,j+jb,j),lda, handle));
                cnrtQueueSync(queue);
            }
        }

    // printf("after transpose, d_output:\n");
    // for(int i = 0; i < 2; i++)
    // {
    //     for(int j = 0; j < lda; j++)
    //     {
    //         for(int h = 0; h < lda; h++)
    //         {
    //             cnrtMemcpy(work_space_h, d_output+i*lda*lda+j*lda+h, sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST);
    //             printf("%8.3f",*work_space_h);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }


    
    printf("before finally, transpose:\n");
    cnrtMemcpy(work_space_h, d_output, sizeof(float)*lda*lda*2, CNRT_MEM_TRANS_DIR_DEV2HOST);
    printf("real result:\n");
    for(int j = 0; j < lda; j++)
    {
        for(int h = 0; h < lda; h++)
        {
            printf("%8.3f",work_space_h[j*lda+h]);
        }
        printf("\n");
    }
    printf("\n");
    printf("imag result:\n");
    for(int j = 0; j < lda; j++)
    {
        for(int h = 0; h < lda; h++)
        {
            printf("%8.3f",work_space_h[lda*lda+j*lda+h]);
        }
        printf("\n");
    }

        // CHECK_RETURN("mluOpCholesky",
        //             sgemm(batch_size, false,true,row-j-jb,jb,j,-1.0f,1.0f,
        //                 OFFSET_ROW(d_output,j+jb,0),lda,stride,
        //                 OFFSET_ROW(d_output,j,0),lda,stride,
        //                 OFFSET_ROW(d_output,j+jb,j),lda,stride, handle));
        //         cnrtQueueSync(queue);

        // cnrtMemcpy(work_space_h, d_output, sizeof(float)*lda*lda*2, CNRT_MEM_TRANS_DIR_DEV2HOST);
        // for(int i = 0; i < 2; i++)
        // {
        //     for(int j = 0; j < lda; j++)
        //     {
        //         for(int h = 0; h < lda; h++)
        //         {
        //             // cnrtMemcpy(work_space_h, d_output+i*lda*lda+j*lda+h, sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST);
        //             printf("%8.3f",work_space_h[i*lda*lda+j*lda+h]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        CHECK_RETURN("mluOpCholesky",
                    transpose(batch_size,2,size_a*size_a,d_output,workspace,handle));
        cnrtQueueSync(queue);
        CNRT_CHECK(cnrtMemcpy(d_output, workspace, type_size*size_a*lda*batch_size, CNRT_MEM_TRANS_DIR_DEV2DEV));

        // printf("after transpose, d_a:\n");

        // for(int j = 0; j < lda; j++)
        // {
        //     for(int h = 0; h < lda; h++)
        //     {
        //         cnrtMemcpy(work_space_h, d_output+j*lda*2+h*2, sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST);
        //         cnrtMemcpy((work_space_h+1), d_output+j*lda*2+h*2+1, sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST);
        //         printf("%8.3f,%8.3f ",work_space_h[0],work_space_h[1]);
        //     }
        //     printf("\n");
        // }
    }
        

    // printf("matrix after calculate:\n");
    //     for(int i = 0; i < batch_size; i++)
    //     {
    //         printf("batch %d:\n",i);
    //         for(int j = 0; j < size_a; j++)
    //         {
    //             for(int k = 0; k < size_a; k++)
    //             {
    //                 cnrtMemcpy(work_space_h, d_output + i*stride+j*lda+k, sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST);
    //                 printf("%.2f ",work_space_h[0]);
    //             }
    //             printf("\n");
    //         }
    //     }
    

    

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