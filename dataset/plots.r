options(rgl.useNULL = TRUE)
library(rgl)
library(shapes)
library(R.matlab)


myplotshapes <- function (A, B = 0, joinline = c(1, 1), orthproj = c(1, 2), color = 1, 
                          symbol = 19) {
    CHECKOK <- TRUE
    if (is.array(A) == FALSE) {
        if (is.matrix(A) == FALSE) {
            cat("Error !! argument should be an array or matrix \n")
            CHECKOK <- FALSE
        }
    }
    if (CHECKOK) {
        k <- dim(A)[1]
        m <- dim(A)[2]
        kk <- k
        if (k >= 15) {
            kk <- 1
        }
        par(pty = "s")
        if (length(c(B)) != 1) {
            par(mfrow = c(1, 2))
        }
        if (length(dim(A)) == 3) {
            A <- A[, orthproj, ]
        }
        if (is.matrix(A) == TRUE) {
            a <- array(0, c(k, 2, 1))
            a[, , 1] <- A[, orthproj]
            A <- a
        }
        out <- defplotsize2(A)
        width <- out$width
        if (length(c(B)) != 1) {
            if (length(dim(B)) == 3) {
                B <- B[, orthproj, ]
            }
            if (is.matrix(B) == TRUE) {
                a <- array(0, c(k, 2, 1))
                a[, , 1] <- B[, orthproj]
                B <- a
            }
            ans <- defplotsize2(B)
            width <- max(out$width, ans$width)
        }
        n <- dim(A)[3]
        lc <- length(color)
        lt <- k * m * n/lc
        color <- rep(color, times = lt)
        lc <- length(symbol)
        lt <- k * m * n/lc
        symbol <- rep(symbol, times = lt)
        plot(A[, , 1], xlim = c(out$xl, out$xl + width), 
             ylim = c(out$yl, out$yl + width), 
             type = "n", xlab = " ", ylab = " ", axes = FALSE)
        for (i in 1:n) {
            select <- ((i - 1) * k * m + 1):(i * k * m)
            points(A[, , i], pch = symbol[select], col = color[select])
            lines(A[joinline, , i])
        }
        if (length(c(B)) != 1) {
            A <- B
            if (is.matrix(A) == TRUE) {
                a <- array(0, c(k, 2, 1))
                a[, , 1] <- A
                A <- a
            }
            out <- defplotsize2(A)
            n <- dim(A)[3]
            plot(A[, , 1], xlim = c(ans$xl, ans$xl + width), 
                 ylim = c(ans$yl, ans$yl + width), type = "n", 
                 xlab = " ", ylab = " ")
            for (i in 1:n) {
                points(A[, , i], pch = symbol[select], col = color[select])
                lines(A[joinline, , i])
            }
        }
    }
}


setwd("~/Documents/github/NestedGrassmann/dataset")

png('digit3_ex.png', width = 10, height = 10, units = "cm", res = 320)
par(mar = c(0, 0, 0, 0))   
myplotshapes(digit3.dat[,,4], joinline = 1:13, symbol = 19)
dev.off() 

png('gorf_ex.png', width = 10, height = 10, units = "cm", res = 320)
par(mar = c(0, 0, 0, 0)) 
myplotshapes(gorf.dat[,,4], joinline = c(1,6,7,8,2,3,4,5,1), symbol = 19)
dev.off() 

png('gorm_ex.png', width = 10, height = 10, units = "cm", res = 320)
par(mar = c(0, 0, 0, 0)) 
myplotshapes(gorm.dat[,,4], joinline = c(1,6,7,8,2,3,4,5,1), symbol = 19)
dev.off() 


