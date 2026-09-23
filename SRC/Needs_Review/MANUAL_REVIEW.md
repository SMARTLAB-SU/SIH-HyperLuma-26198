# Manual Review Register

Complete these items before publishing or submitting the repository.

1. Replace `TeamName` and `SIHID` in the folder name, README, presentation, and documentation; add institution and member details.
2. Add the authorized dataset link, access instructions, license, ethics information, per-class counts, augmentation statistics, averages, and authoritative class descriptions.
3. Reconcile the classification report's three study classes with the eight abbreviated labels stored in `best.pt.encrypted`: `w1.n`, `w2.dys`, `w3.ca`, `w4.ev`, `w5.ec`, `w6.ero`, `w7.d`, and `w8.s`.
4. Confirm whether the detection label `Dysplaisa` is intentionally spelled that way. It is preserved exactly as stored in `best.pt`.
5. Reconcile the detection presentation's 4,125-image, 11-class, 70:20:10 narrative with its conflicting BCC/SK/AK table reporting 847 annotated and 2,069 augmented images.
6. Add training notebooks and training scripts, or explicitly confirm that they are unavailable. None were found in either source repository.
7. Capture at least three de-identified application screenshots. No standalone screenshots were found.
8. Add a detection demonstration video. Only a classification demo video was found.
9. Confirm the classification encrypted-checkpoint password distribution process and whether a non-encrypted `.pth`/`.pt` release artifact is permitted. Only `best.pt.encrypted` was supplied.
10. Verify all reported model metrics against reproducible evaluation runs. The merged documentation transcribes source-reported results and does not claim independent recomputation.
11. Confirm ownership, patient/privacy safeguards, and redistribution rights for models, lookup tables, logos, executable installers, documents, and third-party dependencies before accepting the MIT license for public release.
12. Install Git LFS before the first commit. The classification installer exceeds GitHub's ordinary file-size limit, and several model/executable assets are large.
13. Perform a clean-machine test of both source applications and both installers, including icons, logo display, model lookup paths, encrypted-model handling, camera permissions, and output saving.

No ambiguous source file required relocation into this folder. This register and the inventory identify unresolved metadata, evidence, and reproducibility issues.

