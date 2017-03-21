package de.mpicbg.mosaic.apr;

import java.io.*;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Collectors;

/**
 * Created by ulrik on 11/14/2016.
 */
public class Loader {

    static boolean nativesReady = false;

    public static void loadNatives() throws IOException {
        if(nativesReady) {
            return;
        }

        String lp = System.getProperty("java.library.path");
        File tmpDir = Files.createTempDirectory("apr-natives-tmp").toFile();

        String[] jars = System.getProperty("java.class.path").split(File.pathSeparator);

        for (String s : jars) {
            if (!(s.contains("apr") && s.contains("natives"))) {
                continue;
            }

            try {
                JarFile jar = new JarFile(s);
                Enumeration<JarEntry> enumEntries = jar.entries();

                while (enumEntries.hasMoreElements()) {
                    JarEntry entry = enumEntries.nextElement();

                    // only extract library files
                    String extension = entry.getName().substring(entry.getName().lastIndexOf('.') + 1);
                    if (!(extension.startsWith("so") || extension.startsWith("dll") || extension.startsWith("dylib") || extension.startsWith("jnilib"))) {
                        continue;
                    }

                    File f = new File(tmpDir.getAbsolutePath() + File.separator + entry.getName());

                    if (entry.isDirectory()) {
                        f.mkdir();
                        continue;
                    }

                    InputStream ins = jar.getInputStream(entry);
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    FileOutputStream fos = new FileOutputStream(f);

                    byte[] buffer = new byte[1024];
                    int len;

                    while ((len = ins.read(buffer)) > -1) {
                        baos.write(buffer, 0, len);
                    }

                    baos.flush();
                    fos.write(baos.toByteArray());

                    fos.close();
                    baos.close();
                    ins.close();
                }

                System.setProperty("java.library.path", lp + File.pathSeparator + tmpDir.getCanonicalPath());
            } catch (IOException e) {
                System.err.println("Failed to extract native libraries: " + e.getMessage());
                e.printStackTrace();
            }
        }

        lp = System.getProperty("java.library.path");
        System.setProperty("java.library.path", lp + File.pathSeparator + new java.io.File( "." ).getCanonicalPath() + File.separator + "src" + File.separator + "natives");

        try {
            Field fieldSysPath = ClassLoader.class.getDeclaredField("sys_paths");
            fieldSysPath.setAccessible(true);
            fieldSysPath.set(null, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            System.err.println("Failed to set java.library.path: " + e.getMessage());
            e.printStackTrace();
        }

        try {
            System.loadLibrary("apr");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Unable to load native library: " + e.getMessage());
            String osname = System.getProperty("os.name");
            String osclass = osname.substring(0, osname.indexOf(' ')).toLowerCase();

            System.err.println("Did you include apr-natives-" + osclass + " in your dependencies?");
            System.err.println("java.library.path=" + System.getProperty("java.library.path"));

            throw e;
        }

        nativesReady = true;
    }
}
