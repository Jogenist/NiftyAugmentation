# NiftyAugmentation
Script to apply various randomized augmentations to nifty files and their corresponding segmentation files


Aktueller Stand:
1. Skript lädt alle nii image files aus dem Sample Data Ordner in ein Array (Img_dataset)
2. Skript lädt alle nii label files aus dem Smaple Data Ordner in ein Array (Lab_dataset)
3. es wird eine zufällige Integerzahl generiert mit welcher dann aus einer Whitelist die entsprechende Augmentation ausgewählt wird
4. dann wird die "augmentation" Funktion aufgerufen (hier wird die zufällig generierte Integerzahl sowie die Nummer der zu augmentierenden Nii files (aus dem Image & Label array) übergeben
5. abhängig der zufälligen Integerzahl wird entsprechend augmentiert
6. und das augmentierte nii (sowie das augmentierte Label) im "Sample Data"-Ordner abgespeichert
7. der Name des augmentierten abgespeicherten nii-Files setzt sich aus dem Namen des Original-Files sowie der angewandten Augmentation zusammen


To-Do:
1. "Skew"-Augmentation einbauen
2. "elastic"-Distortion einbauen
3. mehrfache Augmentierungen ermöglichen (z.B. Kombination aus rotation und flip) (aber abschaltbar!)
4. Evtl. Möglichkeiten überlegen mehrere "gecroppte" Nii-Files zu einem neuen Nii zusammenzusetzen
(z.B. 4 Niftys vierteln und dann zu einem neuen zusammensetzen)

erledigte To-Do's:
-  Rotationsache zufällig wählen (bisher wird immer über y gedreht) 
- Möglichkeit wahlweise Augmentations aus dem "Pool" der augmentation-Funktion zu entfernen oder hinzuzufügen -> gelöst mittels Whitelist
- Zufallsfaktor in die einzelnen Augmentierungen reinbringen (z.B. Winkel bei rotation) -> random float für Winkel
- Möglichkeit in die augmentation-Funktion einbringen, dass gar nicht augmentiert wird -> falls nicht augmentiert wird, wird die Funktion verlassen und kein neues nii gespeichert
- 
Known Issues:
1. Der Datensatz den wir am 29.04 bekommen haben enthält einzelne Label-Nii-Files die nur eine dimension von 3 haben (normal sind 4) dadurch können die nii-files nicht alle zusammen in ein array geladen werden. Gelöst hab ich es jetzt dadurch, dass ich die files mit dimension = 3 aus dem Sample Data Ordner gelöscht habe. Der Sample Data Ordner der hier hochgeladen ist, enthält jetzt nur noch die Files die funktionieren.
