args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop(
    "usage: Rscript alice_worker.R <input.tsv> <output.tsv> ",
    "<alpha|beta|humanTRA|humanTRB> <olga-compute_pgen>",
    call. = FALSE
  )
}

input_path <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
output_path <- normalizePath(args[[2]], winslash = "/", mustWork = FALSE)
olga_exe <- Sys.which(args[[4]])
if (!nzchar(olga_exe)) {
  olga_exe <- normalizePath(args[[4]], winslash = "/", mustWork = TRUE)
}

chain_key <- tolower(args[[3]])
chain <- switch(
  chain_key,
  alpha =, tra =, humantra = "humanTRA",
  beta =, trb =, humantrb = "humanTRB",
  stop("chain must be alpha/beta/humanTRA/humanTRB", call. = FALSE)
)

suppressPackageStartupMessages({
  library(data.table)
  library(tcrgrapher)
})

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
metadata_path <- paste0(output_path, ".meta.tsv")

write_empty <- function(status, n_input, n_model_eligible = 0L, n_d_ge_2 = 0L) {
  fwrite(
    data.table(
      clone_key = character(),
      D = integer(),
      p = numeric(),
      q = numeric()
    ),
    output_path,
    sep = "\t"
  )
  fwrite(
    data.table(
      status = status,
      chain = chain,
      n_input = as.integer(n_input),
      n_model_eligible = as.integer(n_model_eligible),
      n_D_ge_2 = as.integer(n_d_ge_2),
      n_output = 0L
    ),
    metadata_path,
    sep = "\t"
  )
}

input <- fread(input_path)
required <- c("clone_key", "cdr3nt", "cdr3aa", "bestVGene", "bestJGene")
missing <- setdiff(required, names(input))
if (length(missing)) {
  stop("missing input columns: ", paste(missing, collapse = ", "), call. = FALSE)
}
if (anyNA(input[, ..required]) || anyDuplicated(input$clone_key)) {
  stop(
    "clone_key and ALICE input columns must be non-missing; ",
    "clone_key must be unique",
    call. = FALSE
  )
}

input[, count := 1L]
setcolorder(input, c("count", "cdr3nt", "cdr3aa", "bestVGene", "bestJGene"))

if (!nrow(input)) {
  write_empty("empty_input", 0L)
  quit(save = "no", status = 0L)
}

# First_release shells out to olga-compute_pgen.  This process-local override
# keeps the official single-core flow while passing Windows-safe arguments.
fixed_parallel_wrapper <- local({
  executable <- olga_exe
  function(DT, cores = 1, chain = "mouseTRB", stats = "OLGA", model = "-") {
    if (cores != 1L || stats != "OLGA") {
      stop("alice_worker supports only cores=1 and stats=OLGA", call. = FALSE)
    }
    if (!("ind" %in% names(DT))) {
      data.table::set(DT, j = "ind", value = seq_len(nrow(DT)))
    }

    input_tmp <- tempfile("alice_olga_", fileext = ".tsv")
    output_tmp <- tempfile("alice_olga_out_", fileext = ".tsv")
    on.exit(unlink(c(input_tmp, output_tmp), force = TRUE), add = TRUE)

    write.table(
      as.data.frame(
        DT[, c("cdr3aa", "bestVGene", "bestJGene", "ind"), with = FALSE]
      ),
      quote = FALSE,
      row.names = FALSE,
      col.names = FALSE,
      sep = "\t",
      file = input_tmp
    )

    if (model != "-") {
      chain <- model
    }
    command_args <- c(
      paste0("--", chain),
      "--display_off",
      "--time_updates_off",
      "--seq_in", "0",
      "--v_in", "1",
      "--j_in", "2",
      "-d", "tab",
      "-i", shQuote(input_tmp, type = "cmd"),
      "-o", shQuote(output_tmp, type = "cmd")
    )
    log <- suppressWarnings(
      system2(executable, command_args, stdout = TRUE, stderr = TRUE)
    )
    status <- attr(log, "status")
    if (is.null(status)) {
      status <- 0L
    }
    if (status != 0L || !file.exists(output_tmp)) {
      stop(
        "OLGA failed (status ", status, "): ",
        paste(tail(log, 20L), collapse = "\n"),
        call. = FALSE
      )
    }

    stats_output <- fread(output_tmp, header = FALSE)
    if (nrow(stats_output) != nrow(DT) || ncol(stats_output) < 2L) {
      stop("OLGA output shape does not match its input", call. = FALSE)
    }
    DT[, Pgen := as.numeric(stats_output[[2]])]
    DT
  }
})
assignInNamespace(
  x = "parallel_wrapper_beta",
  value = fixed_parallel_wrapper,
  ns = "tcrgrapher"
)

prepared_input <- tempfile("alice_patient_", fileext = ".tsv")
fwrite(input, prepared_input, sep = "\t")
object <- TCRgrapher(
  prepared_input,
  match("count", names(input)),
  match("cdr3nt", names(input)),
  match("cdr3aa", names(input)),
  match("bestVGene", names(input)),
  match("bestJGene", names(input))
)
unlink(prepared_input, force = TRUE)

# Reproduce only the official pre-OLGA eligibility check so zero-row samples
# return a canonical empty file rather than tripping a package stopifnot.
namespace <- asNamespace("tcrgrapher")
model_name <- if (chain == "humanTRA") "OLGAVJ_HUMAN_TRA" else "OLGAVJ_HUMAN_TRB"
model_vj <- get(model_name, envir = namespace)
eligible <- clonoset(object)
eligible <- eligible[
  !grepl(cdr3aa, pattern = "*", fixed = TRUE) &
    ((nchar(cdr3nt) %% 3L) == 0L) &
    count >= 0L
]
eligible <- eligible[
  bestVGene %in% rownames(model_vj) &
    bestJGene %in% colnames(model_vj)
]
n_model_eligible <- nrow(eligible)
if (n_model_eligible) {
  eligible <- get("calculate_nb_of_neighbors", envir = namespace)(eligible)
}
n_d_ge_2 <- if (n_model_eligible) sum(eligible$D >= 2L) else 0L

if (!n_d_ge_2) {
  write_empty(
    "no_D_ge_2",
    n_input = nrow(input),
    n_model_eligible = n_model_eligible,
    n_d_ge_2 = 0L
  )
  quit(save = "no", status = 0L)
}

# First_release constructs all three model entries eagerly, including the
# unused mouse object that is absent from the trimmed human-only snapshot.
# A local copy with a harmless placeholder preserves the upstream function
# body while allowing the selected human TRA/TRB branch to run unchanged.
alice_pipeline <- ALICE_pipeline
compatibility_environment <- new.env(parent = environment(alice_pipeline))
compatibility_environment$OLGAVJ_MOUSE_TRB <- matrix(
  0,
  nrow = 1L,
  ncol = 1L,
  dimnames = list("unused" = "unused", "unused" = "unused")
)
environment(alice_pipeline) <- compatibility_environment

result_object <- alice_pipeline(
  object,
  Q_val = 27,
  cores = 1,
  thres_counts = 0,
  N_neighbors_thres = 2,
  p_adjust_method = "BH",
  chain = chain,
  stats = "OLGA",
  model = "-"
)
result <- clonoset(result_object)[
  ,
  .(
    clone_key,
    D = as.integer(ALICE.D),
    p = ALICE.p_value,
    q = ALICE.p_adjust
  )
]
setorder(result, clone_key)
fwrite(result, output_path, sep = "\t")
fwrite(
  data.table(
    status = "ok",
    chain = chain,
    n_input = nrow(input),
    n_model_eligible = n_model_eligible,
    n_D_ge_2 = n_d_ge_2,
    n_output = nrow(result)
  ),
  metadata_path,
  sep = "\t"
)
