from requirements import *

@register_keras_serializable(package='MyLosses')
# CCA loss for DCCA model
def cca_loss():
    # Keras loss functions must only take (y_true, y_pred) as parameters
    def inner_cca_objective(y_true, y_pred):
        r1 , r2 , eps = 10**-3 , 10** -3 , 10** -9
        o1 = o2 = int (y_pred.shape [1] // 2)
        # unpack ( separate ) the output of networks for view 1 and view 2
        H1 = tf . transpose ( y_pred [: , 0: o1 ])
        H2 = tf . transpose ( y_pred [: , o1 : o1 + o2 ])
        m = tf . shape ( H1 ) [1]

        H1bar = H1 - tf . cast ( tf . divide (1 , m) , tf . float32 ) * tf . matmul (H1 , tf .
        ones ([ m , m ]) )
        H2bar = H2 - tf . cast ( tf . divide (1 , m) , tf . float32 ) * tf . matmul (H2 , tf .
        ones ([ m , m ]) )

        SigmaHat12 = tf . cast ( tf . divide (1 , m - 1) , tf . float32 ) * tf . matmul (
        H1bar , H2bar , transpose_b = True )
        SigmaHat11 = tf . cast ( tf . divide (1 , m - 1) , tf . float32 ) * tf . matmul (
        H1bar , H1bar , transpose_b = True ) + r1 * tf . eye ( o1 )
        SigmaHat22 = tf . cast ( tf . divide (1 , m - 1) , tf . float32 ) * tf . matmul (
        H2bar , H2bar , transpose_b = True ) + r2 * tf . eye ( o2 )
        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1 , V1 ] = tf . linalg . eigh ( SigmaHat11 )
        [D2 , V2 ] = tf . linalg . eigh ( SigmaHat22 )
        
        # get eigen values that are larger than eps
        posInd1 = tf . where ( tf . greater (D1 , eps ))
        D1 = tf . gather_nd ( D1 , posInd1 )
        V1 = tf . transpose ( tf . nn . embedding_lookup ( tf . transpose ( V1 ) , tf . squeeze (
        posInd1 )))

        posInd2 = tf . where ( tf . greater (D2 , eps ))
        D2 = tf . gather_nd ( D2 , posInd2 )
        V2 = tf . transpose ( tf . nn . embedding_lookup ( tf . transpose ( V2 ) , tf . squeeze (
        posInd2 )))

        SigmaHat11RootInv = tf . matmul ( tf . matmul ( V1 , tf . linalg . diag ( D1 ** -0.5)
        ) , V1 , transpose_b = True )
        SigmaHat22RootInv = tf . matmul ( tf . matmul ( V2 , tf . linalg . diag ( D2 ** -0.5)
        ) , V2 , transpose_b = True )

        Tval = tf . matmul ( tf . matmul ( SigmaHat11RootInv , SigmaHat12 ) ,
        SigmaHat22RootInv )
        corr = tf . sqrt ( tf . linalg . trace ( tf . matmul ( Tval , Tval , transpose_a = True )
        ))
        return - corr
    return inner_cca_objective

