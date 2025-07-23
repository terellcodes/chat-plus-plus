import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Chat++ | PDF RAG IDE",
  description: "An intelligent chat application with advanced RAG strategies for PDF analysis",
  keywords: ["RAG", "PDF", "AI", "Chat", "Document Analysis"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#1e1e1e] text-[#d4d4d4] h-screen overflow-hidden`}
      >
        {children}
      </body>
    </html>
  );
}
