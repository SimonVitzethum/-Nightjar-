/* GPU timeline recording for the decode loop. No-ops unless built with HAVE_CUPTI and run
 * with QWEN_CUPTI=1. See qwen35_cupti.cu for what the numbers mean. */
#ifndef QWEN35_CUPTI_H
#define QWEN35_CUPTI_H
#ifdef __cplusplus
extern "C" {
#endif

#if defined(HAVE_CUPTI) && !defined(QWEN_NO_CUDA)
void q35cupti_init(void);
void q35cupti_reset(void);
void q35cupti_report(int ntok);
int  q35cupti_on(void);
#else
static inline void q35cupti_init(void){}
static inline void q35cupti_reset(void){}
static inline void q35cupti_report(int ntok){ (void)ntok; }
static inline int  q35cupti_on(void){ return 0; }
#endif

#ifdef __cplusplus
}
#endif
#endif
