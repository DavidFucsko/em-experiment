# fastLink (Enamorado, Fifield & Imai 2019): EM over all pairs of history.csv x later.csv in the directory given first,
# with a Beta prior on the match share: prior mean and weight given second and third (fastLink's prior.lambda, w.lambda).
# Every further argument is one field as name:kind:cut_a:cut_p, kind jw | lv | number | exact (fastLink's gammaCKpar,
# gammaNUMCKpar, gammaKpar). Writes parameters.csv: per field and level (gamma) the estimated m and u, and the match
# share as the field "match_share".
suppressMessages(library(fastLink))
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
prior.lambda <- as.numeric(args[2])
w.lambda <- as.numeric(args[3])
specs <- strsplit(args[-(1:3)], ":")
read <- function(side) read.csv(file.path(dir, paste0(side, ".csv")), colClasses = "character", na.strings = "")
A <- read("history")
B <- read("later")
cores <- max(1, parallel::detectCores() - 1)

gamma <- function(spec) {
  a <- as.matrix(A[[spec[1]]])
  b <- as.matrix(B[[spec[1]]])
  cut.a <- as.numeric(spec[3])
  cut.p <- as.numeric(spec[4])
  switch(spec[2],
         jw = gammaCKpar(a, b, n.cores = cores, cut.a = cut.a, cut.p = cut.p, method = "jw", w = 0.1),
         lv = gammaCKpar(a, b, n.cores = cores, cut.a = cut.a, cut.p = cut.p, method = "lv"),
         number = gammaNUMCKpar(matrix(as.numeric(a)), matrix(as.numeric(b)), n.cores = cores, cut.a = cut.a, cut.p = cut.p),
         exact = gammaKpar(a, b, n.cores = cores))
}

counts <- tableCounts(lapply(specs, gamma), nobs.a = nrow(A), nobs.b = nrow(B), n.cores = cores)
em <- emlinkMARmov(counts, nobs.a = nrow(A), nobs.b = nrow(B), varnames = sapply(specs, `[`, 1),
                   prior.lambda = prior.lambda, w.lambda = w.lambda)
patterns <- as.data.frame(em$patterns.w)
parameters <- do.call(rbind, lapply(seq_along(specs), function(k) {
  levels <- sort(unique(na.omit(patterns[[k]])))
  stopifnot(length(levels) == length(em$p.gamma.k.m[[k]]))
  data.frame(field = specs[[k]][1], gamma = levels, m = em$p.gamma.k.m[[k]], u = em$p.gamma.k.u[[k]])
}))
parameters <- rbind(parameters, data.frame(field = "match_share", gamma = NA, m = em$p.m, u = NA))
write.csv(parameters, file.path(dir, "parameters.csv"), row.names = FALSE)
cat("fastLink: match share", em$p.m, "after", em$iter.converge, "iterations\n")
